import os, sys
sys.path.append('..')
import yaml
import torch
from models.ODE_model import FeatureClassifier
from evaluation.arcface import iresnet50
from models.model import OcclusionAwareGenerator
from dataset.frames_dataset import FramesDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from tqdm import tqdm, trange
from sync_batchnorm import DataParallelWithCallback
# from torchsummary import summary
import warnings
warnings.filterwarnings('ignore')


def nowarp_inference(generator, source):
    output_dict = {}
    out = generator.first(source)
    for i in range(len(generator.down_blocks)):
        out = generator.down_blocks[i](out)
    output_dict['encoder_output'] = out
    out = generator.bottleneck(out)
    for i in range(len(generator.up_blocks)):
        out = generator.up_blocks[i](out)
    out = generator.final(out)
    out = F.sigmoid(out)
    output_dict['prediction'] = out
    return output_dict


def feature_inference(generator, feature):
    out = generator.bottleneck(feature)
    for i in range(len(generator.up_blocks)):
        out = generator.up_blocks[i](out)
    out = generator.final(out)
    out = F.sigmoid(out)
    return out


if __name__ == '__main__':
    # define feature classifer
    feature_classifer = FeatureClassifier()
    feature_classifer = DataParallelWithCallback(feature_classifer, device_ids=[0])


    # load pretrained face model and fomm encoder
    face_classifier = iresnet50(False, fp16=True)
    face_ckpt = torch.load('../ckpt/face_model/backbone.pth')
    face_classifier.load_state_dict(face_ckpt)
    face_classifier.cuda()
    face_classifier.eval()
    config_path = '../config/vox-256.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                       **config['model_params']['common_params'])
    generator_ckpt = torch.load('../ckpt/backbone.pth')
    generator.load_state_dict(generator_ckpt['generator'])
    generator.cuda()
    generator.eval()

    batch_size = 12
    epochs = 100
    # define dataset
    dataset = FramesDataset(is_train=True, **config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # define loss
    CE_loss = torch.nn.CrossEntropyLoss()
    L1_loss = torch.nn.L1Loss()

    # define optimizer
    optim = torch.optim.Adam(feature_classifer.parameters(), lr=0.00001)


    iter = 0
    true_label = torch.ones(batch_size, dtype=torch.long).cuda()
    fake_label = torch.zeros(batch_size, dtype=torch.long).cuda()
    for epoch in trange(epochs):
        for i, data in enumerate(dataloader):
            out = nowarp_inference(generator, data['source'].cuda())
            feature_true = out['encoder_output'].detach()
            out = nowarp_inference(generator, out['prediction'].cuda())
            feature_fake = out['encoder_output'].detach()
            loss = 0
            logits_fake = feature_classifer(feature_fake)
            logits_true = feature_classifer(feature_true)
            loss += CE_loss(logits_true, true_label)
            loss += CE_loss(logits_fake, fake_label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            iter = iter + 1
            if iter % 100 == 0:
                print(loss.item())
                with torch.no_grad():
                    logits_fake = F.softmax(feature_classifer(feature_fake))
                    logits_true = F.softmax(feature_classifer(feature_true))
                    print('the true lable predicted:', logits_fake)
                    print('the fake lable predicted:', logits_true)
    torch.save({'classifer': feature_classifer.state_dict()}, '../ckpt/classifier.pth')
















