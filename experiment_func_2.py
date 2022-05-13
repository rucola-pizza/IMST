""" 
Experiment functions
    실험에 필요한 대부분의 기능을 제공할 예정, 실험이 종료되면 utils 로 합쳐지거나 기능에 따라서 개별 모듈들로 분리 
Todo:
    ** add_hook function에서 hook을 걸 range 설정 
    ** hook을 통해 얻은 feature 를 reshape, permute 해서 사용가능한 형태로 저장하는 함수 

"""
from unittest.mock import NonCallableMagicMock
from matplotlib import scale
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from PIL import Image
import skimage.io
import skimage 
import numpy as np 
import cv2
import scipy
from scipy.linalg import eigh
from scipy import ndimage
import matplotlib.pyplot as plt 

from custom_datasets import ImageDataset
from LOST.object_discovery import lost 

import torchvision 
import dino.vision_transformer as vits

def crop(img):
    pass


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
        _arch = "vit" + arch[4:] 
        model = vits.__dict__[_arch](patch_size = patch_size, num_classes = 0)
    else:
        print('No arch selected')
        return 
    
    #Gradient 계산 멈춤 
    for p in model.parameters():
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
    img = datasets.dataloader[0][0]  # c x w x h-
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

def get_feature(model, input_img, cuda=True):
    """
    Input:
        model - pre-trained model on cuda 
        input_img - image after img_prop 
            c x w x h
    Output:
        cls_list - cls from all layer
        output_list
        q_list 
        k_list 
        v_list
    """
    # model input 형태는 batch x c x w x h 이므로 unsqueeze 해준다
    input_img = input_img.unsqueeze(0)
    if cuda:
        return model.get_features(input_img.to('cuda'))
    else:
        return model.get_features(input_img)

def lost_output(feats, dims, scales, init_image_size, k_patches=100):
    """
    Input:
        feat - q or k or v
            1(batch) x Num_token x hidden dimension
        dims - [w_featmap, h_featmap]
            w_featmap, h_featmap은 img_prop을 거치면 나옴 
        scales - [patch_size, patch_size]
        init_image_size - 원본 이미지 크기, dataloader에서 로딩한 이미지의 크기 
        k_patches - seed expansion 갯수 
    Output: 
        pred
        A
        labeled_array
        scores
        seed 
    """
    return lost(feats=feats, dims=dims, scales=scales, init_image_size=init_image_size, k_patches=k_patches)

#LOST Result visualize 
def lost_visualize(image, pred, A, scores, labeled_array , seed, scales, dims, plot_seed):
    """
    Input:
        image - scikit image 
        pred - bbox 
        A - Sim matrix 
        scores - patch scores 
        seed - seed patch number 
        scales - [patch_size, patch_size]
        dims - [w_featmap, h_featmap]
    Output:
        image - original image with bbox & seed 
        corr - selected patches 
        deg - inverse degree 
        labeled_array - connected component 
    """
    w_featmap, h_featmap = dims 
    #Plot box 
    cv2.rectangle(
        image,
        (int(pred[0]) , int(pred[1])),
        (int(pred[2]) , int(pred[3])),
        (255, 0, 0), 3,
    )

    if plot_seed:
        print('plot seed')
        s_ = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
        size_ = np.asarray(scales) /2
        cv2.rectangle(
            image,
            (int(s_[1] * scales[1] - (size_[1] / 2)), int(s_[0] * scales[0] - (size_[0] / 2))),
            (int(s_[1] * scales[1] + (size_[1] / 2)), int(s_[0] * scales[0] + (size_[0] / 2))),
            (0, 255 ,0), -1, 
        )
    
    #Plot fms 
    binA = A.cpu().numpy().copy()
    binA[binA < 0] = 0
    binA[binA > 0 ] =1
    
    im_corr = np.zeros((3, len(scores))) 
    where = binA[seed,: ] > 0
    im_corr[:, where] = np.array([128/255, 133/ 255, 133/ 255]).reshape((3,1))
    im_corr[:, seed] = [204/255, 37/255, 41/255]
    im_corr = im_corr.reshape(3, w_featmap, h_featmap)
    im_corr = (
        nn.functional.interpolate(
            torch.from_numpy(im_corr).unsqueeze(0),
            scale_factor = scales,
            mode = "nearest",
        )[0].cpu().numpy()
    )
    
    corr  = im_corr.transpose((1,2,0))
    
    im_deg = (
        nn.functional.interpolate(
            torch.from_numpy(1 / binA.sum(-1)).reshape(1, 1, w_featmap, h_featmap),
            scale_factor=scales,
            mode="nearest",
        )[0][0].cpu().numpy()
    )
    
    return image, corr, im_deg, labeled_array

def tokencut_output(feats, k, cls, dims, scales, init_image_size, pos_patch_idx = None, tau = 0, eps = 1e-5, im_name='', no_binary_graph=False, A=None):

    """
    Input: 
        feats : k or q or v
            기존 tokencut code 에선 feature 와 cls 토큰 분리를 본 함수에서 해줬으나 lost 와 통일을 위해서 분리해준 feature
            가 들어온다고 가정한다. 
        dims - [w_featmap, h_featmap]
        scales - [patch_size, patch_size]
        init_image_size
        tau - threshold for graph construction
        eps
        im_name 
    """
    init_image_size = init_image_size[1:]
    print(f"input cls shape : {cls.shape}")
    cls = cls.squeeze(0)
    feats = feats.squeeze(0)
    print(f"cls shape : {cls.shape}")
    print(f"feats shaep : {feats.shape}")
    print(f"key shape : {k.shape}")
    
    sim = F.cosine_similarity(cls, k, dim=-1)
    print(f"sim shape : {sim.shape}")
    value = torch.sort(sim, descending=True)[0][0]
    print(value.shape)
    pos_value = value[torch.where(value > 0 )[0]]
    mean_pos_value = torch.mean(pos_value)
    max_len = len(torch.where(value > mean_pos_value)[0])
    pos_patch_idx = list(torch.sort(sim, descending=True)[1][0, :max_len].cpu().numpy())
    

    if A is None:
        feats = feats.squeeze(0)
        print(f"Tokencut feats shape : {feats.shape}")
        feats = F.normalize(feats, p=2)
        A = (feats @ feats.transpose(1,0))
        A = A.cpu().numpy()
    else:
        A = A
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])

    # Using average point to compute bipartition 
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    
    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects,cc = detect_box(bipartition, seed, dims, pos_patch_idx=pos_patch_idx, scales=scales, initial_im_size=init_image_size) ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1

    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)

#TokenCut detect box 
def detect_box(bipartition, seed, dims, initial_im_size=None, pos_patch_idx= None ,scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    
    pos_patch_idx : List 
    
    """
    seeds = []
    seeds.append(seed)
    seeds = seeds + pos_patch_idx
    pred_feats = []
    preds = []
    cc = []
    objects, num_objects = ndimage.label(bipartition) 

    for i in range(len(seeds)):
        x_, y_ = np.unravel_index(seeds[i], dims)
        cc_ = objects[x_, y_]
        if cc_ == 0:
            pass 
        else: 
            cc.append(cc_)
    cc = np.unique(cc)
    

    if principle_object:
        for cc_ in cc:
            mask = np.where(objects == cc_)
            # 너무 작은 box 가 선택되면 스킵한다
            if len(mask[0]) < 4:
                continue  

       # Add +1 because excluded max
            ymin, ymax = min(mask[0]), max(mask[0]) + 1
            xmin, xmax = min(mask[1]), max(mask[1]) + 1
            # Rescale to image size
            r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
            r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
            pred = [r_xmin, r_ymin, r_xmax, r_ymax]
            preds.append(pred)
         
            # Check not out of image size (used when padding)
            if initial_im_size:
                pred[2] = min(pred[2], initial_im_size[1])
                pred[3] = min(pred[3], initial_im_size[0])
        
        # Coordinate predictions for the feature space
        # Axis different then in image space
            pred_feat = [ymin, xmin, ymax, xmax]
            pred_feats.append(pred_feat)

        return preds, pred_feats, objects, mask
    else:
        raise NotImplementedError

def tokencut_visualize(img, eigvec, preds, labeled_array , dims, scales, seed, plot_seed=False):
    #second smallest eigvec 
    eigvec = scipy.ndimage.zoom(eigvec, scales, order=0, mode='nearest')
    w_featmap, h_featmap = dims
    #image and pred
    image = np.copy(img)
    #plot the box 
    print(preds)
    for pred in preds:
        cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,
        )
    if plot_seed:
        print('plot seed')
        s_ = np.unravel_index(seed, (w_featmap, h_featmap))
        size_ = np.asarray(scales) /2
        cv2.rectangle(
            image,
            (int(s_[1] * scales[1] - (size_[1] / 2)), int(s_[0] * scales[0] - (size_[0] / 2))),
            (int(s_[1] * scales[1] + (size_[1] / 2)), int(s_[0] * scales[0] + (size_[0] / 2))),
            (0, 255 ,0), -1, 
        )
 

    return image, eigvec , labeled_array


def lost_tokencut_visualize(img,
                            eigvec, tokencut_pred, tokencut_labeled_array,tokencut_seed,
                            lost_pred, lost_sim_matrix, lost_labeled_array ,lost_scores, lost_seed,
                            scales, dims, plot_seed = True,
                            cmap='cividis'):
    lost_image_input = img.copy()
    tokencut_image_input = img.copy()
    
    #LOST visualize output 
    lost_image, _, lost_deg, lost_labeled_array = lost_visualize(
        image=lost_image_input, 
        pred=lost_pred, 
        A = lost_sim_matrix, 
        scores=lost_scores,
        labeled_array=lost_labeled_array,
        seed=lost_seed,
        scales=scales,
        dims=dims,
        plot_seed=plot_seed
    )
    
    #Tokencut visualize output
    tokencut_image, tokencut_eigvec, tokencut_labeled_array = tokencut_visualize(
        img=tokencut_image_input,
        eigvec=eigvec,
        preds=tokencut_pred,
        labeled_array=tokencut_labeled_array,
        dims=dims,
        scales=scales,
        plot_seed=plot_seed,
        seed=tokencut_seed
    )
    
    plt.subplot(231)
    plt.imshow(lost_image)
    plt.title("LOST")

    plt.subplot(232)
    plt.imshow(lost_deg, cmap=cmap)

    plt.subplot(233)
    plt.imshow(lost_labeled_array, cmap='tab20')
    
    plt.subplot(234)
    plt.imshow(tokencut_image)
    plt.title('Seong Taek Cut')

    plt.subplot(235)
    plt.imshow(tokencut_eigvec, cmap=cmap)
    
    plt.subplot(236)
    plt.imshow(tokencut_labeled_array, cmap='tab20')
    
    plt.tight_layout()


def lost_tokencut(img_path, model, patch_size, feature, depth,
                  tau = 0.2, no_binary_graph = False, A=None,plot_seed = False,
                  cmap='cividis'):
    scikit_img = load_image(img_path=img_path)
    # scikit_img = skimage.transform.resize(scikit_img,
    #                                        (480,480),
    #                                        )
    single_tensor_image = img_ImageDataset(img_path=img_path)
    init_img_size = single_tensor_image.shape
    scales = [patch_size, patch_size]
    input_image, w_featmap, h_featmap = img_prop(single_tensor_image, patch_size=patch_size)
    dims = [w_featmap, h_featmap]
    _, output_list, q_list, k_list, v_list = model.get_features(input_image.unsqueeze(0).to('cuda'))
    
    if feature == 'k':
        feat = k_list[depth][:, 1:, :]
    elif feature == 'q':
        feat = q_list[depth][:, 1:, :]
    elif feature == 'v':
        feat = v_list[depth][:, 1:, :]
    elif feature == 'output':
        feat = output_list[depth][:, : ,:]
    else:
        print('Wrong featrue, feature must be one of [k, q, v, output]')
    
    tokencut_pred, tokencut_labeled_array, _, tokencut_seed, _, tokencut_eigvec = tokencut_output(
        feats = feat,
        cls=q_list[-1][: , 0 , :],
        k = k_list[-1][: , 1: , ],
        dims=dims,
        scales=scales,
        init_image_size=init_img_size,
        tau = tau,
        no_binary_graph=no_binary_graph,
        A=A
    )

    lost_pred, lost_sim_matrix, lost_labeled_array, lost_scores, lost_seed = lost_output(
        feats = feat,
        dims = dims,
        scales=scales,
        init_image_size=init_img_size
    )
    #print("lost seed : {}".format(lost_seed))
    #print("TokenCut seed : {}".format(tokencut_seed))
    lost_tokencut_visualize(img=scikit_img,
                            eigvec=tokencut_eigvec,
                            tokencut_pred=tokencut_pred,
                            tokencut_labeled_array=tokencut_labeled_array,
                            tokencut_seed=tokencut_seed,
                            lost_pred = lost_pred,
                            lost_sim_matrix=lost_sim_matrix,
                            lost_labeled_array=lost_labeled_array,
                            lost_scores=lost_scores,
                            lost_seed=lost_seed,
                            scales=scales,
                            dims = dims,
                            cmap=cmap,
                            plot_seed=plot_seed
                            )
    
    return tokencut_labeled_array, lost_labeled_array, tokencut_pred
    
def cls_feature_sim_matrix(cls, feats, dims):
    
    w_featmap, h_featmap = dims
    sim = (feats@cls.transpose(1,0)).squeeze()
    sim_matrix = sim.reshape(w_featmap, h_featmap)
    
    if type(sim_matrix) == torch.Tensor:
        if sim_matrix.device.type == 'cuda':
            sim_matrix = sim_matrix.cpu().numpy()
    
    return sim_matrix

def attention_visualize(img_path, model, patch_size):
    """
    Input:
        input_image - from img_prop 
    """
    
    img = ImageDataset(img_path = img_path)
    input_image, w_featmap, h_featmap = img_prop(img=img, patch_size=patch_size)
    
    attn = model.get_last_selfattention(input_image.unsqueeze(0).to('cuda'))
    nh = attn.shape[1]
    attn = attn[0, : , 0 , 1:].reshape(nh, w_featmap, h_featmap)
    
    
    