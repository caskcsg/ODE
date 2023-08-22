import os, sys
sys.path.append('..')
import yaml
from time import gmtime, strftime
from tqdm import tqdm, trange
from argparse import ArgumentParser
from utils.util import *

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from dataset.frames_dataset import SeqFramesDataset
from models.encoder import *
from models.bottleneck import *
from models.decoder import *
from models.fuser import MLF
from models.model import KPDetector, DenseMotionNetwork
from models.model import find_models, GeneratorFullModel
from models.fuser import MLF
from utils.logger import Logger
from torchvision import models

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='../config/mlf_hwt.yaml', help='path to config')
    parser.add_argument('--checkpoint', help='path to load ckpt when resume')
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    time_now = time.strftime('%Y-%m-%d-%H-%M')
    log_dir = os.path.join('../log_f', time_now)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = Logger(log_dir=log_dir)
    logger.log_config(config)

    device = torch.device('cuda', 0)

    # 数据集
    train_params = config['train_params']
    dataset = SeqFramesDataset(**config['dataset_params'])
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], drop_last=True, sampler=sampler)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=train_params['batch_size'], drop_last=False, num_workers=train_params['num_workers'])

    # 生成器模型
    encoder = find_models(**config['model_params']['encoder'])
    decoder = find_models(**config['model_params']['decoder'])
    bottleneck = find_models(**config['model_params']['bottleneck'])
    kp_detector = KPDetector(**config['model_params']['kp_detector'])
    dense_motion = DenseMotionNetwork(**config['model_params']['dense_motion'])
    model = GeneratorFullModel(kp_detector, dense_motion, encoder, bottleneck, decoder,
                               discriminator=None, train_params=train_params, mode='single')
    model = torch.load(config['train_params']['ckpt_path'])
    if "reconstruction" not in model.loss_weights:
        model.loss_weights['reconstruction'] = 0
    model.to(device)
    model.eval()

    # fusion模型
    multilevel_fusion = MLF(**config['model_params']['multilevel_fusion'])
    multilevel_fusion.to(device)
    multilevel_fusion.train()

    # 优化器和学习率策略
    optim = Adam(multilevel_fusion.parameters(), lr=train_params['learning_rate'])
    scheduler = MultiStepLR(optim, milestones=train_params['step_milestones'], gamma=train_params['gamma'])

    # perceptual loss
    vgg = Vgg19().to(device)

    # 训练
    iter = 0
    for epoch in trange(train_params['num_epochs']):
        for repeat in range(train_params['num_repeats']):
            for x in tqdm(dataloader):
                # put data to device
                for k, v in x.items():
                    if torch.is_tensor(v):
                        x.update({k: v.to(device)})
                kp_source = model.kp_extractor(x['source'])
                kp_driving = model.kp_extractor(x['driving'])
                # kp_driving_prev = model.kp_detector(x['driving_prev'])
                # make texture features (direct)
                dense_motion = model.dense_motion(source_image=x['source'], kp_driving=kp_driving,
                                                 kp_source=kp_source)
                encode_features = model.encoder(x['source'])
                deformed_features = model.deform_input(encode_features, dense_motion['deformation'])
                if dense_motion['occlusion_map'].shape[-1] != deformed_features.shape[-1]:
                    occlusion_map = F.interpolate(dense_motion['occlusion_map'], size=deformed_features.shape[2:],
                                                  mode='bilinear')
                else:
                    occlusion_map = dense_motion['occlusion_map']
                deformed_features = deformed_features * occlusion_map
                texture_features = deformed_features
                # make pose features (reconstruct)
                driving_prev = model.decoder(model.bottleneck(model.encoder(x['driving_prev'])))
                kp_driving_prev = model.kp_extractor(driving_prev)
                dense_motion = model.dense_motion(source_image=driving_prev, kp_driving=kp_driving,
                                                 kp_source=kp_source)
                encode_features = model.encoder(driving_prev)
                deformed_features = model.deform_input(encode_features, dense_motion['deformation'])
                if dense_motion['occlusion_map'].shape[-1] != deformed_features.shape[-1]:
                    occlusion_map = F.interpolate(dense_motion['occlusion_map'], size=deformed_features.shape[2:],
                                                  mode='bilinear')
                else:
                    occlusion_map = dense_motion['occlusion_map']
                deformed_features = deformed_features * occlusion_map
                pose_features = deformed_features
                # feature fusing
                fused_features = multilevel_fusion(texture_features, pose_features)
                generated = model.decoder(model.bottleneck(fused_features))
                losses = {}
                if 'feature' in config['train_params']['loss_weights']:
                    # feature level supervision
                    gt_features = model.encoder(x['driving'])
                    losses['feature'] = nn.functional.l1_loss(fused_features, gt_features) * config['train_params']['loss_weights']['feature']

                if 'reconstruction' in config['train_params']['loss_weights']:
                    losses['recon'] = nn.functional.l1_loss(generated, x['driving']) * config['train_params']['loss_weights']['reconstruction']

                if 'perceptual' in config['train_params']['loss_weights']:
                    pyramide_real = model.pyramid(x['driving'])
                    pyramide_generated = model.pyramid(generated)
                    value_total = 0
                    for scale in config['train_params']['scales']:
                        x_vgg = model.vgg(pyramide_generated['prediction_' + str(scale)])
                        y_vgg = model.vgg(pyramide_real['prediction_' + str(scale)])

                        for i, weight in enumerate(config['train_params']['loss_weights']['perceptual']):
                            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                            value_total += config['train_params']['loss_weights']['perceptual'][i] * value
                    losses['per'] = value_total

                loss_values = [val.mean() for val in losses.values()]
                loss = sum(loss_values)

                optim.zero_grad()
                loss.backward()
                optim.step()
                iter = iter + 1

                if iter % train_params['iters_save_img'] == 0:
                    img = torch.cat([x['source'][0], x['driving'][0], generated[0]], dim=-1)
                    logger.log_loss(losses, {'iter': iter, 'epoch': epoch})
                    logger.log_img(img, iter=iter)
        scheduler.step()
        if (epoch+1) % train_params['epochs_save_ckpt'] == 0:
            logger.log_ckpt(multilevel_fusion, epoch)



