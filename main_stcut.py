"""
main file for evaluation

TODO

"""
import os 
import argparse
import pickle
from requests import patch

import torch 
import torch.nn as nn 
import numpy as np 
import datetime

from tqdm import tqdm
from PIL import Image
from yaml import parse
from TokenCut.datasets import Dataset, ImageDataset, bbox_iou

from experiment_func_2 import * 
from object_discovery import stcut 

import matplotlib.pyplot as plt 
import time 

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ST Cut")
    parser.add_argument(
        "--arch",
        default="dino_small",
        type=str,
        choices=[
            "dino_base", "dino_small"
        ]
    )

    parser.add_argument(
        "--patch_size", default=16, type=int
    )

    parser.add_argument(
        "--dataset", 
        default="VOC07",
        type=str,
        choices=[
            None, "VOC07", "VOC12", "COCO20k"
        ]
    )
    
    parser.add_argument(
        "--set", 
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"]
    )

    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directorcy store prediction and visualization"
    )
    
    parser.add_argument(
        "--tau", default=0.2, type=float, help="TokenCut의 tau값"
    )
    parser.add_argument(
        "--eps", default=1e-5, type=float, help="Eps for defining the Graph"
    )
    
    #Evaluation arg
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    
    #STCut arg
    parser.add_argument(
        "--depth", default=-1, type=int, help="사용한 feature를 추출한 Layer의 깊이"
    )
    parser.add_argument(
        "--min_box_size", default=5, type=int, help="Pos seed 가 속해있는 box 중 pred로 간주하기 위한 최소 patch 갯수"
    )
    
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------------------------------------
    # Dataset Code 
    dataset = Dataset(args.dataset, args.set, args.no_hard)
    
    # -------------------------------------------------------------------------------------------------------
    # Mdoel 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(arch=args.arch, patch_size=args.patch_size, device=device)
    
    # -------------------------------------------------------------------------------------------------------
    # Directories 
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------------------------------------
    # Naming 
    
    exp_name = f"STCut-{args.arch}"
    exp_name += f"patch{args.patch_size}_Layer{args.depth}"
    
    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    num_gt = 0
    cnt = 0 
    TP = 0 
    FP = 0
    
    start_time = time.time()
    pbar = tqdm(dataset.dataloader)
    
    scales = [args.patch_size, args.patch_size]
    
    for im_id, inp in enumerate(pbar):
        
        img = inp[0]

        init_image_size = img.shape
        
        im_name = dataset.get_image_name(inp[1])
        #im_name => None 이면 gt 가 없다는 뜻 
        if im_name == None:
            continue
        
        #padded image, dims 
        padded_img, w_featmap, h_featmap = img_processing(img=img, patch_size=args.patch_size, device=device)
        
        #Ground Truth 
        gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)
        #Annotation이 없는경우 continue, VOC07, VOC12에서 발생가능 
        if gt_bbxs is not None:
            num_gt += len(gt_bbxs)
            if gt_bbxs.shape[0] == 0 and args.no_hard:
                continue
        

        # -------------------------------------------------------------------------------------------------------
        # Feature Extraction 
        with torch.no_grad():
            
            # cls_tokens, patch_token, query, key, value from all layers 
            # 일단은 query, key 만 사용 
            _, _, q_list, k_list, _ = model.get_features(padded_img.unsqueeze(0).to(device))
            
            feat = k_list[args.depth][:, 1:, :]
            cls_ = q_list[args.depth][:, 0, :]
            
            feat_sim = k_list[-1][:, 1: , :]
            cls_sim = q_list[-1][: , 0 , :]
            
            preds, _, _, _ = stcut(feat=feat, 
                                    cls = cls_,
                                    feat_sim = feat_sim,
                                    cls_sim = cls_sim,
                                    dims=[w_featmap, h_featmap], 
                                    scales = scales,
                                    min_box_size=args.min_box_size,
                                    init_image_size = init_image_size,
                                    tau = args.tau,
                                    im_name = im_name,
                                    eps = args.eps,                                                          
                                    )
        

        # -------------------------------------------------------------------------------------------------------
        # Some nice eval    
        for pred in preds:
            ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
            if torch.any(ious >= 0.5):
                TP += 1 
            else:
                FP += 1 
        
        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"corloc : {round(100*TP/(TP+FP),4)}% | TP:{TP} | FP:{FP} | Num GT : {num_gt}")
    
    end_time = time.time()
    
    print(f"Length of Dataset : {len(dataset.dataloader)}")
    print(f"Counted image : {cnt}")
    print('-'*30)
    print(f"Experiment Settings")
    print(f'Datasets : {args.dataset}')
    print(f'Architecture : {args.arch}')
    print(f"Patch size : {args.patch_size}")
    print(f"Min box size : {args.min_box_size}")
    print(f"Number of GT BBX : {num_gt}")
    print(f"Finding Box : {TP+FP}  |  TP : {TP}  |  FP : {FP}")
    print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')
    print(f"Final CorLoc : {round(100*TP/(TP+FP),4)}")       
    print('-'*30)

        
        


    
    