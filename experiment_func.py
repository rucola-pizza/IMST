import torch 
import torch.nn as nn 

import torchvision 
import dino.vision_transformer as vits

#load model from hub 
#DINO만 가능, BEiT, iBOT, MAE 추가필요 
def get_model(arch, patch_size, device):
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
