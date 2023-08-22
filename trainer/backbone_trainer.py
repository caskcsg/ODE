import os, sys
sys.path.append('..')
import yaml
from time import gmtime, strftime
from tqdm import tqdm, trange
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")

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
from models.model import KPDetector, DenseMotionNetwork
from models.model import find_models, GeneratorFullModel
from utils.logger import Logger




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='../config/backbone_res_EBD_ted.yaml', help='path to config')
    parser.add_argument('--checkpoint', help='path to load ckpt when resume')
    parser.add_argument('--local_rank', default=-1, type=int)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # logger
    time_now = time.strftime('%Y-%m-%d-%H-%M')
    log_dir = os.path.join('../log', time_now)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = Logger(log_dir=log_dir)
    logger.log_config(config)

    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(opt.local_rank)
    # device = torch.device('cuda', opt.local_rank)
    device = torch.device('cuda', 0)

    train_params = config['train_params']
    dataset = SeqFramesDataset(**config['dataset_params'])
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], drop_last=True, sampler=sampler)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], drop_last=True, num_workers=train_params['num_workers'])

    encoder = find_models(**config['model_params']['encoder'])
    decoder = find_models(**config['model_params']['decoder'])
    bottleneck = find_models(**config['model_params']['bottleneck'])
    kp_detector = KPDetector(**config['model_params']['kp_detector'])
    dense_motion = DenseMotionNetwork(**config['model_params']['dense_motion'])
    model = GeneratorFullModel(kp_detector, dense_motion, encoder, bottleneck, decoder,
                               discriminator=None, train_params=train_params, mode='single')
    model.to(device)
    # model = DistributedDataParallel(model, device_ids=[opt.local_rank])

    optim = Adam(model.parameters(), lr=train_params['learning_rate']['encoder'])
    scheduler = MultiStepLR(optim, milestones=train_params['step_milestones'], gamma=train_params['gamma'])
    # optim_encoder = Adam(encoder.parameters(), lr=train_params['learning_rate']['encoder'])
    # scheduler_encoder = StepLR(optim_encoder, step_size=train_params['step_size'], gamma=train_params['gamma'])
    # optim_bottleneck = Adam(bottleneck.parameters(), lr=train_params['learning_rate']['bottleneck'])
    # scheduler_bottleneck = StepLR(optim_bottleneck, step_size=train_params['step_size'], gamma=train_params['gamma'])
    # optim_decoder = Adam(decoder.parameters(), lr=train_params['learning_rate']['decoder'])
    # scheduler_decoder = StepLR(optim_encoder, step_size=train_params['step_size'], gamma=train_params['gamma'])
    # optim_kp_detector = Adam(kp_detector.parameters(), lr=train_params['learning_rate']['kp_detector'])
    # scheduler_kp_detector = StepLR(optim_kp_detector, step_size=train_params['step_size'], gamma=train_params['gamma'])
    # optim_dense_motion = Adam(dense_motion.paramters(), lr=train_params['learning_rate']['dense_motion'])
    # scheduler_dense_motion = StepLR(optim_dense_motion, step_size=train_params['step_size'], gamma=train_params['gamma'])

    iter = 0
    for epoch in trange(train_params['num_epochs']):
        for repeat in range(train_params['num_repeats']):
            for x in tqdm(dataloader):
                for k, v in x.items():
                    if torch.is_tensor(v):
                        x.update({k: v.to(device)})
                losses, generated = model(x)

                loss_values = [val.mean() for val in losses.values()]
                loss = sum(loss_values)

                optim.zero_grad()
                loss.backward()
                optim.step()
                iter = iter + 1

                if iter % train_params['iters_save_img'] == 0:
                    img = torch.cat([x['source'][0], x['driving'][0], generated['prediction'][0]], dim=-1)
                    logger.log_loss(losses, {'iter':iter, 'epoch': epoch})
                    logger.log_img(img, iter=iter)
        scheduler.step()
        if epoch % train_params['epochs_save_ckpt'] == 0:
            logger.log_ckpt(model, epoch)






