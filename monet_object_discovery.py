from tabnanny import verbose
from typing import ChainMap
from imageio import save
import torch 
import faiss 
import cv2 
import time 
import numpy as np 
from einops import einops
from tqdm import tqdm 
from scipy import ndimage
from monet_utils import * 
import matplotlib.pyplot as plt 
from pycocotools.coco import COCO

def vector_sim(vec1, vec2):
    return vec1 @ vec2

def sim_matrix(vec1, vec2):
    return vec1 @ vec2.transpose(1,0)

def seed_patch_idx(cls_sim, seed_type = 'pos', top_k = None):
    if seed_type == 'pos':
        return torch.where(cls_sim > 0)[0]
    elif seed_type == 'top_k'and top_k is not None:
        return torch.sort(cls_sim, descending=True)[1][:top_k]
    else:
        raise('Check input param')

def seed_expansion_idx(qk_matrix, seed_patch, expansion_type = 'pos', top_k=None):
    if expansion_type == 'pos':
        return torch.where(qk_matrix[seed_patch] > 0)[1].unique()
    elif expansion_type == 'mean_pos':
        if top_k is None:
            pos = torch.where(torch.sum(qk_matrix[seed_patch], dim=0) > 0)[0]
            return torch.sort(torch.sum(qk_matrix[seed_patch], dim=0), descending=True)[1][:len(pos)]
        else:
            pos = torch.where(torch.sum(qk_matrix[seed_patch], dim=0) > 0)[0]
            return torch.sort(torch.sum(qk_matrix[seed_patch], dim=0), descending=True)[1][:top_k]
    else:
        raise('Check input param')

def key_matrix_idx(kk_matrix, seed_expansion, sel_type = 'mean_pos'):
    if sel_type == 'mean_pos':
        pos = torch.where(torch.sum(kk_matrix[seed_expansion], dim=0) > 0)[0]
        return torch.sort(torch.sum(kk_matrix[seed_expansion], dim=0) , descending=True)[1][:len(pos)]
    else:
        raise('Check input param')
    
def head_selec(query, key, depth, n_head,
               seed_type, seed_top_k,
               expansion_type, expansion_top_k,
               sel_type,scale,
               left, right,
               plot=False, dims=None):
    query_cls = query[0, 0 ,:]
    query_feature = query[0, 1:, :]
    key_feature = key[0, 1: , :]
    
    #Head split 
    query_cls = einops.rearrange(query_cls, '(n h) -> n h', n=n_head)
    query_feature = einops.rearrange(query_feature, 't (n h) -> n t h', n=n_head)
    key_feature = einops.rearrange(key_feature, 't (n h) -> n t h', n=n_head)
    cls_sim = torch.einsum('nth, nh -> nt', key_feature, query_cls)
    qk_matrix = torch.einsum('nah, nch -> nac', query_feature, key_feature)
    kk_matrix = torch.einsum('nah, nch -> nac', key_feature, key_feature)

    cum_mask = torch.zeros(query_feature.shape[1])
    save_mask = torch.zeros((n_head, cls_sim.shape[-1]))
    for i in range(n_head):
        seed = seed_patch_idx(cls_sim[i], seed_type=seed_type, top_k=seed_top_k)
        seed_expansion = seed_expansion_idx(qk_matrix=qk_matrix[i], seed_patch=seed,
                                            expansion_type=expansion_type, top_k=expansion_top_k)
        object_idx = key_matrix_idx(kk_matrix=kk_matrix[i], seed_expansion=seed_expansion,
                                 sel_type=sel_type)
        
        cum_mask[object_idx] += 1 
        save_mask[i][object_idx] = 1
        
        if plot:
            plt.figure()
            plt.imshow(save_mask[i].cpu().numpy().reshape(dims))
    
    #Head select
    if scale:
        mean_ = torch.mean(cum_mask)
        std_ = torch.std(cum_mask)
        cum_mask = (cum_mask - mean_) / std_
    
    if plot:
        plt.figure()
        plt.imshow(cum_mask.cpu().numpy().reshape(dims))
    
    weighted_mask = save_mask * cum_mask
    patch_sum = torch.sum(weighted_mask, dim=-1)
    head_sort = torch.sort(patch_sum)[1]
    head_ = head_sort[left:right]

    cls = einops.rearrange(query_cls[head_], 'n h -> (n h)')
    query_feature = einops.rearrange(query_feature[head_], 'n d h -> d (n h)')
    key_feature = einops.rearrange(key_feature[head_],'n d h -> d (n h)' )

    return cls, query_feature, key_feature

def mask_gen(query, key, depth, n_head,
               seed_type, seed_top_k,
               expansion_type, expansion_top_k,
               sel_type,scale,
               left, right,
               return_all=False, dims=None, plot=False):

    cls, query_feature, key_feature = head_selec(query=query,
                                                 key=key,
                                                 depth=depth,
                                                 n_head=n_head,
                                                 seed_type=seed_type, seed_top_k=seed_top_k,
                                                 expansion_type=expansion_type, expansion_top_k=expansion_top_k,
                                                 sel_type=sel_type, scale=scale,
                                                 left=left, right=right,
                                                 plot=plot, dims=dims
                                                 )
    
    cls_sim = vector_sim(key_feature, cls)
    qk_matrix = sim_matrix(query_feature, key_feature)
    kk_matrix = sim_matrix(key_feature, key_feature)
    seed = seed_patch_idx(cls_sim=cls_sim,
                          seed_type=seed_type,
                          top_k=seed_top_k)
    seed_expansion = seed_expansion_idx(qk_matrix=qk_matrix,
                                        seed_patch=seed,
                                        expansion_type=expansion_type,
                                        top_k=expansion_top_k)
    object_idx = key_matrix_idx(kk_matrix=kk_matrix,
                                seed_expansion=seed_expansion,
                                sel_type=sel_type,)
    mask = torch.zeros(cls_sim.shape[-1])
    mask[object_idx] = 1
    if return_all:
        return seed, seed_expansion, object_idx, mask, cls, query_feature, key_feature
    return mask 

def kmeans_refine(feature, mask, dims, 
                  min_cluster, max_cluster, random_state,max_iter,
                  ratio, thred, cosine, gpu,
                  cc=False, return_all=False, plot=False,
                  ):
    feature = feature.cpu().numpy()
    d = feature.shape[-1]
    num_patch = feature.shape[0]
    result = np.zeros(num_patch)
    if not cc:  
        for n_cluster in range(min_cluster, max_cluster + 1):
            kmeans = faiss.Kmeans(d=d,
                                k=n_cluster,
                                niter = max_iter,
                                spherical = cosine,
                                verbose=False,
                                gpu = gpu,
                                max_points_per_centroid = 1000000,
                                seed=random_state)
            kmeans.train(feature)
            _, pred = kmeans.index.search(feature,1)
            pred = pred.flatten() + 1 

            if plot:
                tmp = np.zeros(dims[0]*dims[1])
                for pred_ in np.unique(pred):
                    clst = np.zeros(num_patch)
                    where = np.where(pred == pred_)[0] 
                    clst[where] = 1
                    if np.sum(clst*mask) / len(where) >= ratio:
                        result[where] += 1
                        tmp[where] += 1 
                elementwise = mask * pred 
                plt.figure(figsize=(20,12))
                plt.subplot(131)
                plt.imshow(pred.reshape(dims), cmap='tab20')
                plt.subplot(132)
                plt.imshow(np.clip(elementwise, 0.1, 0.9).reshape(dims))
                plt.subplot(133)
                plt.imshow(tmp.reshape(dims))
            else:
                for pred_ in np.unique(pred):
                    clst = np.zeros(num_patch)
                    where = np.where(pred == pred_)[0] 
                    clst[where] = 1
                    if np.sum(clst*mask) / len(where) >= ratio:
                        result[where] += 1
                
        # result = result / (max_cluster - min_cluster +1)
        # thred_result = np.where(result >= thred, 1, 0)
    else:
        for n_cluster in range(min_cluster, max_cluster + 1):
            kmeans = faiss.Kmeans(d=d,
                                k=n_cluster,
                                spherical = cosine,
                                verbose=False,
                                gpu = gpu,
                                max_points_per_centroid = 1000000,
                                seed=random_state)
            kmeans.train(feature)
            _, pred = kmeans.index.search(feature,1)
            pred = pred.reshape(dims)
            pred, _ = ndimage.label(pred)

            for pred_ in np.unique(pred):
                clst = np.zeros(num_patch)
                where = np.where(pred == pred_)[0] 
                clst[where] = 1
                if np.sum(clst*mask) / len(where) >= ratio:
                    result[where] += 1
            
    # result = result / (max_cluster - min_cluster +1)
    if len(np.unique(result)) == 1:
        if return_all:
            return mask, mask
        else:
            return mask  
    result = (result - result.min()) / (result.max() - result.min())
    thred_result = np.where(result >= thred, 1, 0)
    
    
    if return_all:
        return result, thred_result
    return thred_result                            
        
         
def object_selec(mask, cls, query, key, 
                 dims,
                 obj_select_type, top_k,
                 min_box_size=5):
    
    object_idx = np.where(mask > 0)[0]
    query = query[object_idx]
    key = key[object_idx]
    seg_mask = np.zeros(len(mask))
    cc, _ = ndimage.label(mask.reshape(dims))
    cc = cc.flatten()

    if obj_select_type == 'cls_sim':
        cls_sim = key@cls
        if top_k is None:   
            cls_sim = object_idx[torch.where(cls_sim > 0)[0].cpu().numpy()]
            selected_cc = np.unique(cc[cls_sim])
        else:
            cls_sim = object_idx[torch.sort(cls_sim, descending=True)[1][:top_k].cpu().numpy()]    
            selected_cc = np.unique(cc[cls_sim])
    
    elif obj_select_type == 'kk_matrix' and top_k is not None:
        kk_matrix = key@key.transpose(1,0)
        kk_sum = object_idx[torch.sort(torch.sum(kk_matrix, dim=1), descending=True)[1][:top_k].cpu().numpy()]
        selected_cc = np.unique(cc[kk_sum])
    
    elif obj_select_type == 'kk_cls_sim':
        kk_matrix = key@key.transpose(1,0)
        cls_sim = key@cls
        cls_sim = torch.where(cls_sim > 0 )[0]
        kk_matrix = kk_matrix[cls_sim]
        kk_sum = object_idx[torch.where(torch.sum(kk_matrix, dim=0) > 0)[0].cpu().numpy()]
        selected_cc = np.unique(cc[kk_sum])
    
    elif obj_select_type == 'all':
        selected_cc = np.unique(cc)
    
    
    #No CC Selected
    if len(selected_cc) == 0:
        print('No Selected CC')
        selected_cc = np.unique(cc)
    
    box_sizes = []   
    for cc_ in selected_cc:
        box_sizes.append(len(np.where(cc == cc_)[0]))
    min_box_size = np.min([np.max(box_sizes), min_box_size])
    for idx,cc_ in enumerate(selected_cc):
        cc_cor = np.where(cc == cc_)[0]
        if len(cc_cor) < min_box_size or cc_ == 0:
            continue
        seg_mask[np.where(cc == cc_)[0]] = idx+1
    # print(f'Num of Object : {len(np.unique(seg_mask)) - 1}')
    
    return seg_mask
        

        
        

            


        


        
        
        
        
        
    



        
    
    

