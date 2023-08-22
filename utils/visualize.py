import cv2
import torch
import os
from utils.flow_viz import flow_to_image
import numpy as np
from torch.nn.functional import normalize

def save_middle_results(image, type, save_path):
    if type == 'mask':
        _, mask_nums, H, W = image.shape
        temp_mask = image[0].permute(1,0,2)
        temp_mask = temp_mask.reshape(H, mask_nums * W)
        temp_mask = (temp_mask * 255).detach().cpu().numpy().astype('uint8')
        cv2.imwrite(os.path.join(save_path, 'mask.jpg'), temp_mask)
    elif type == 'sparse_deformed':
        _, deform_nums, C, H, W = image.shape
        temp_deform = image[0].permute(1, 2, 0, 3)
        temp_deform = temp_deform.reshape(C, H, deform_nums*W)
        temp_deform = (temp_deform * 255).permute(1, 2, 0).detach().cpu().numpy().astype('uint8')
        cv2.imwrite(os.path.join(save_path, 'sparse_deformed.jpg'), temp_deform)
    elif type == 'occlusion_map':
        _, _, H, W = image.shape
        temp_occlusion = (image[0,0,:,:] * 255).detach().cpu().numpy().astype('uint8')
        cv2.imwrite(os.path.join(save_path, 'occlusion.jpg'), temp_occlusion)
    elif type == 'deformation':
        _, H, W, _ = image.shape
        xx = np.linspace(-1, 1, H)[:, np.newaxis].repeat(W, axis=1)[:,:,np.newaxis]
        yy = np.linspace(-1, 1, W)[np.newaxis, :].repeat(H, axis=0)[:,:,np.newaxis]
        standard_flow = np.concatenate([yy,xx], axis=-1)
        flow_img = flow_to_image(image[0].detach().cpu().numpy() - standard_flow)
        # flow_img = flow_to_image(image[0].detach().cpu().numpy())
        cv2.imwrite(os.path.join(save_path, 'optic_flow.jpg'), flow_img)

        pass
    elif type == 'deformed':
        temp_deformed = image[0]
        temp_deformed = (temp_deformed * 255).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
        cv2.imwrite(os.path.join(save_path, 'deformed.jpg'), temp_deformed)
    elif type == 'prediction' or type == 'source' or type == 'driving':
        temp_prediction = image[0]
        temp_prediction = (temp_prediction * 255).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
        cv2.imwrite(os.path.join(save_path, type + '.jpg'), temp_prediction)
    elif type == 'diff' or type == 'error' or type == 'change':
        temp_prediction = abs(image[0])
        temp_prediction = temp_prediction / temp_prediction.max()
        temp_prediction = torch.mean(temp_prediction, dim=0)
        temp_prediction = (temp_prediction * 255 * 2).cpu().detach().numpy().astype('uint8')
        # temp_prediction = (temp_prediction * 255).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
        cv2.imwrite(os.path.join(save_path, type + '.jpg'), temp_prediction)
    # consider as feature map, first normalize to 0-1 then take average
    else:
        # TODO: to show a two polar feature map, not simply abs
        image = image.squeeze()
        if len(image.shape) > 4:
            raise Exception('too much dim (>4)')
        elif len(image.shape) == 4:
            image = torch.abs(image)
            B, C, H, W = image.shape
            image = image.view(B*C, H, W)
            max_value = torch.max(image.view(-1, H * W), dim=-1)
            feature = image/max_value[0].view(-1, 1, 1)
            feature = (torch.mean(feature, dim=0) * 255).cpu().detach().numpy().astype('uint8')
            cv2.imwrite(os.path.join(save_path, type + '.jpg'), feature)
        elif len(image.shape) == 3:
            image = torch.abs(image)
            C, H, W = image.shape
            max_value = torch.max(image.view(C, H*W), dim=-1)
            feature = image/max_value[0].view(-1, 1, 1)
            feature = (torch.mean(feature, dim=0) * 255 * 5).cpu().detach().numpy().astype('uint8')
            cv2.imwrite(os.path.join(save_path, type + '.jpg'), feature)
        elif len(image.shape) == 2:
            image = torch.abs(image)
            feature = image / image.max()
            feature = (feature * 255).cpu().detach().numpy().astype('uint8')
            cv2.imwrite(os.path.join(save_path, type + '.jpg'), feature)



