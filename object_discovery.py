
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

def stcut(feat ,cls, feat_sim, cls_sim, dims, scales, min_box_size, init_image_size, tau, im_name, eps):
    """
    Input
        feat : 
        cls : 
        dims : 
        scales : 
        init_image_size : 
        tau : 
        im_name :  ? 이걸 왜 받는거지 
        eps :
    Output
        preds : 
        objects : 
        mask : 
        seed : 
        eigenvec : 
    """
    # channel x w x h -> w x h 
    init_image_size = init_image_size[1:] 
    
    #squeeze 해서 batch 차원 없애기 
    cls = cls.squeeze(0)

    #find pos patch idx 
    if True: # TMP Delete Here when you fail and come back 
        cls_sim = cls_sim.squeeze(0)
        sim = F.cosine_similarity(cls_sim, feat_sim, dim=-1)
        value = torch.sort(sim, descending=True)[0][0]
        pos_value = value[torch.where(value > 0 )[0]]
        mean_pos_value = torch.mean(pos_value)
        # max_len = len(torch.where(value > mean_pos_value)[0])
        max_len = len(torch.where(value > 0.0 )[0])
        pos_patch_idx = list(torch.sort(sim, descending=True)[1][0, :max_len].cpu().numpy())        
    else: 
        sim = F.cosine_similarity(cls, feat, dim=-1)
        value = torch.sort(sim, descending=True)[0][0]
        pos_value = value[torch.where(value > 0 )[0]]
        mean_pos_value = torch.mean(pos_value)
        max_len = len(torch.where(value > mean_pos_value)[0])
        pos_patch_idx = list(torch.sort(sim, descending=True)[1][0, :max_len].cpu().numpy())
        
    # Token Cut     
    feat = feat.squeeze(0)  # sim 계산할때는 batch 차원이 필요해서 여기서 squeeze 
    feat = F.normalize(feat, p=2)
    A = (feat@feat.transpose(1,0))
    A = A.cpu().numpy()

    A = A > tau 
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    
    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects,cc = detect_box(bipartition=bipartition,
                                     seed=seed, 
                                     pos_patch_idx=pos_patch_idx, 
                                     dims=dims, 
                                     min_box_size=min_box_size,
                                     scales=scales, 
                                     init_image_size=init_image_size) ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1

    return np.asanyarray(pred), objects, mask, seed,  

def detect_box(bipartition, seed, pos_patch_idx, dims, min_box_size, scales, init_image_size ):
    """
    Input
        seed : largest value index from eigenvector 
        dims : [w_featmap, h_featmap]
        scales : [patch_size, patch_size]
        pos_patch_idx : cosin_sim(cls-q , feat-k) 을 수행한 결과중 양의 상관관계 이면서 그 중 평균값을 넘는 패치의  index  
        init_image_size
    Output
        pred : List of pred 
        preds_feats : List of pred in feature space 
        object
        mask
    """

    seeds = [seed]
    seeds = seeds + pos_patch_idx
    pred_feats = []
    preds = []
    cc = []
    objects, _ = ndimage.label(bipartition)
    
    #seed + pos_patch_idx 중에서 Object에 속하는게 있는지 확인하고 그 object number 를 cc 리스트에 저장  
    for idx, seed_ in enumerate(seeds):
        x, y = np.unravel_index(seed_, dims)
        cc_ = objects[x, y]
        if idx ==0:
            tokencut_cc = cc_
        #Object number 가 0 이면 배경으로 취급
        if cc_ == 0:
            continue
        cc.append(cc_)
    cc = np.unique(cc)
    

    for cc_ in cc:
        mask = np.where(objects == cc_)
        #너무 작은 box 는 선택하지 않는다 
        if not cc_ == tokencut_cc:
            if len(mask[0]) <= min_box_size:
                continue
        
        # Add + 1 because excluded max 
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]
        preds.append(pred)
        
        # Check not out of image size (used when padding)
        if init_image_size:
            pred[2] = min(pred[2], init_image_size[1])
            pred[3] = min(pred[3], init_image_size[0])
    
        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feat = [ymin, xmin, ymax, xmax]
        pred_feats.append(pred_feat)
    
    return preds, pred_feats, objects, mask 