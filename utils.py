import torch 
import torch.nn as nn 
import numpy as np 

def img_processing(img, patch_size, device):
    """
    데이터 로더에서 나온 이미지를 받아서 모델에 넣기위해서 padding하고 padding image, w_featmap, h_featmap을 반환 
    Input
        img : img from dataloader 
        patch_size : 8 or 16
        device 
    Output:
        padded image : patch 사이즈에 맞게 빈 부분을 0으로 패딩한 이미지 
        w_featmap : 가로 patch 갯수 
        h_featmap : 새로 patch 갯수 
    """
    #ceil = 
    size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / patch_size) * patch_size),
            int(np.ceil(img.shape[2] / patch_size) * patch_size),
        )
    paded = torch.zeros(size_im)
    paded[:, : img.shape[1], : img.shape[2]] = img
    img = paded

    # # Move to gpu
    if device == torch.device('cuda'):
        img = img.cuda(non_blocking=True)
        # Size for transformers
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    
    return img, w_featmap, h_featmap