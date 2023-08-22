import os, sys
sys.path.append('..')
import yaml
import torch
from models.ODE_model import FeatureClassifier, FeatureFusion
from models.model import OcclusionAwareGenerator, KPDetector
from dataset.frames_dataset import SeqFramesDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.fft import fft2, ifft2
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

# warp feature according to kp_source and kp_driving
def feature_warp(generator, feature, source, kp_source, kp_driving):
    dense_motion = generator.dense_motion_network(source_image = source, kp_driving=kp_driving,
                                                  kp_source=kp_source)
    deformation = dense_motion['deformation']
    occusion_map = dense_motion['occlusion_map']
    out = generator.deform_input(feature, deformation)
    out = out * occusion_map
    return out


def fomm_feature_inference(generator, feature):
    out = generator.bottleneck(feature)
    for i in range(len(generator.up_blocks)):
        out = generator.up_blocks[i](out)
    out = generator.final(out)
    out = F.sigmoid(out)
    return out


def classifier_enhance(feature, classifer, eta=0):
    ce_loss = torch.nn.CrossEntropyLoss()
    feature = feature.detach()
    batch_size = feature.shape[0]
    true_label = torch.ones(batch_size, dtype=torch.long).cuda()
    feature.requires_grad = True
    logits_fake = classifer(feature)
    loss = 0
    loss += ce_loss(logits_fake, true_label)
    grad = torch.autograd.grad(loss, feature)[0].detach()
    grad = grad / grad.max()
    feature_enhanced = feature - eta * feature.max() * grad
    return feature_enhanced


if __name__ == '__main__':
    # load feature classifer and fomm model
    # load feature classifer
    feature_classifer = FeatureClassifier()
    feature_classifer = DataParallelWithCallback(feature_classifer, device_ids=[0])
    feature_ckpt = torch.load('../ckpt/classifier.pth')
    feature_classifer.load_state_dict(feature_ckpt['classifer'])
    feature_classifer.cuda()
    feature_classifer.eval()

    # load fomm model
    fomm_config_path = '../config/vox-256.yaml'
    with open(fomm_config_path) as f:
        fomm_config = yaml.load(f, Loader=yaml.FullLoader)
    fomm_generator = OcclusionAwareGenerator(**fomm_config['model_params']['generator_params'],
                                       **fomm_config['model_params']['common_params'])
    fomm_ckpt = torch.load('../ckpt/FOMM/vox-cpk.pth.tar')
    fomm_generator.load_state_dict(fomm_ckpt['generator'])
    fomm_generator.cuda()
    fomm_generator.eval()

    # load_keypoint_detector
    kp_detector = KPDetector(**fomm_config['model_params']['kp_detector_params'],
                             **fomm_config['model_params']['common_params'])
    kp_detector.load_state_dict(fomm_ckpt['kp_detector'])
    kp_detector.cuda()
    kp_detector.eval()

    # define featurefusion
    fusion = FeatureFusion()
    fusion.cuda()

    batch_size = 12
    epochs = 100
    # define dataset
    dataset = SeqFramesDataset(is_train=True, **fomm_config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define loss
    CE_loss = torch.nn.CrossEntropyLoss()
    L1_loss = torch.nn.L1Loss()

    # define optimizer and scheduler
    optim = torch.optim.Adam(fusion.parameters(), lr=1e-3)
    schedule = StepLR(optim, 10, gamma=0.1)


    iter = 0
    true_label = torch.ones(batch_size, dtype=torch.long).cuda()
    fake_label = torch.zeros(batch_size, dtype=torch.long).cuda()
    for epoch in trange(epochs):
        for i, data in enumerate(dataloader):
            data['source'] = data['source'].cuda()
            data['driving'] = data['driving'].cuda()
            data['driving_prev'] = data['driving_prev'].cuda()
            kp_source = kp_detector(data['source'])
            kp_driving_prev = kp_detector(data['driving_prev'])
            kp_driving = kp_detector(data['driving'])
            feature_single = fomm_generator(data['source'],kp_driving=kp_driving, kp_source=kp_source)['deformed_feature']
            self_reconstructed_driving_prev = fomm_generator(data['driving_prev'], kp_driving=kp_driving, kp_source=kp_source)['prediction']
            degraded_source_feature = nowarp_inference(fomm_generator, self_reconstructed_driving_prev)
            enhanced_feature = classifier_enhance(degraded_source_feature['encoder_output'], classifer=feature_classifer)
            feature_multi = feature_warp(fomm_generator, enhanced_feature, data['driving_prev'],
                                              kp_source=kp_driving_prev, kp_driving=kp_driving)
            feature_fusion = fusion(feature_single, feature_multi)
            pred = fomm_feature_inference(fomm_generator, feature_fusion)
            true = data['driving']
            # true_fft = fft2(true)
            # true_real = true_fft.real
            # true_imag = true_fft.imag
            # true_fft = torch.cat([true_real, true_imag], dim=1)
            # pred_fft = fft2(pred)
            # pred_real = pred_fft.real
            # pred_imag = pred_fft.imag
            # pred_fft = torch.cat([pred_real, pred_imag], dim=1)
            # loss = L1_loss(true_fft, pred_fft)
            loss = L1_loss(true, pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            iter = iter + 1
            if iter % 100 == 0:
                print('iter:',str(iter), '      ', loss.item())
        # schedule.step()
        print('update learning rate:', optim.param_groups[0]['lr'])
    torch.save({'fusion': fusion.state_dict()}, '../ckpt/fusion.pth')
    pass


















