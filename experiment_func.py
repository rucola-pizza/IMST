""" 
Experiment functions
    실험에 필요한 대부분의 기능을 제공할 예정, 실험이 종료되면 utils 로 합쳐지거나 기능에 따라서 개별 모듈들로 분리 
Todo:
    ** add_hook function에서 hook을 걸 range 설정 
    ** hook을 통해 얻은 feature 를 reshape, permute 해서 사용가능한 형태로 저장하는 함수 

"""
import torch 
import torch.nn as nn 
from PIL import Image
import skimage.io
import numpy as np 

from datasets import ImageDataset

import torchvision 
import dino.vision_transformer as vits

#load model from hub 
#DINO만 가능, BEiT, iBOT, MAE 추가필요 
def get_model(arch, patch_size, device):
    """ 
    Input 
        arch 
        patch_size 
        device 
    """
    if "dino" in arch: 
        model = vits.__dict__['vit'](patch_size = patch_size, num_classes = 0)
    else:
        print('No arch selected')
        return 
    
    #Gradient 계산 멈춤 
    for p in model.parameter():
        p.requires_grad = False
    
    #Load pre-train model 
    if arch == "dino_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif arch == "dino_small" and patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
    elif arch == "dino_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif arch == "dino_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    else: 
        print("Not proper arch")
        return None 
    
    #DINO refer 
    state_dict  = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/dino/" + url
    )
    strict_loading = True   #ignore unmatched key
    msg = model.load_state_dict(state_dict, strict = strict_loading)
    print(
        "Pre-trained weights founde at {} and loaded with msg : {}".format(url, msg)
    )
    
    model.eval()    #inference mode 
    model.to(device)
    return model 

#Add hook
def add_qkv_hook(model, depth):
    """
    Input:
        model - target model 
    Output:
        model - model with forward hook
        feat_out - list contains feature 
    """
    feat_out = []
    def hook_fn_forward_qkv(module, input, output, index):
        feat_out[index] = output
        
    for i in range(depth):
        model._modules["blocks"][-i]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        
    return model, feat_out


#load img 
def load_image(img_path, return_img_name = False):

    img_name = img_path.split("/")[-1]
    
    #load img 
    img = skimage.io.imread(img_path)

    #return img or img & img_name 
    if not return_img_name:
        return img
    else:
        return img, img_name

#load ImageDatasets
def load_ImageDataset(img_path):
    datasets = ImageDataset(image_path=img_path)
    return datasets

#img from ImageDatasets
#single image 의 경우 batch 차원을 제거해야 한다
def img_ImageDataset(img_path):
    datasets = load_ImageDataset(img_path)
    img = datasets.dataloader[0][0].unsqueeze(0) #batch x w x h x c -> w x h x c 
    return img

def img_prop(img, patch_size):
    """이미지를 DINO의 인풋으로 만들기위해서 제로패딩 및 사이즈 저장과정
    Args:
        img : img from dataset 
            shape : batch x channel x width x height
    Returns:
        w_featmpa, h_featmap : number of patch width, height 
        padded_img : zero padded image
    """
    #padding the image with zeros
    size_im = (
        img.shape[0],
        int(np.ceil(img.shape[1] / patch_size) * patch_size),
        int(np.ceil(img.shape[2] / patch_size) * patch_size),
    )
    paded = torch.zeros(size_im)
    paded[:, : img.shape[1], : img.shape[2]] = img
    img = paded

    #size for transformer
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    return img, w_featmap, h_featmap
    
def get_features(model, image):
    return None