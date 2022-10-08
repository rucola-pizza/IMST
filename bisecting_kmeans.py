import torch 
import faiss 
import os
import json  
import numpy as np 
import time 
import random
from tqdm import tqdm 
from monet_utils import * 
from monet_object_discovery import *
from monet_datasets import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("BisectingKmeans")
    parser.add_argument(
        "--arch",
        default="dino_base", 
        type=str,
        choices=[
            "dino_small", 
            "dino_base"
        ]
    )
    parser.add_argument(
        "--patch_size", 
        default=8,
        type=int
    )
    parser.add_argument(
        "--dataset",
        default="COCO20k",
        type=str,
        choices=["COCO20k", "VOC12", "VOC07"]
    )
    #Resize (640, 960)
    parser.add_argument(
        "--resize",
        action="store_true"
    )
    #Never use 
    parser.add_argument(
        "--remove_hard",
        action="store_true"
    )
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=[
            "val",
            "train", 
            "trainval", 
            "test"
        ]
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./BiKmeans_centroid",
    )
    
    #Kmeans Param 
    parser.add_argument(
        "--token_path", 
        default='/data_hdd1/batch_token_COCO20k_8_dino_base_output_1.npy',
        type=str
    )
    parser.add_argument(
        '--token_ratio' , 
        default = 1.0,
        type=float
    )
    parser.add_argument(
        "--n_cluster",
        default=10,
        type=int
    )
    parser.add_argument(
        "--n_iter",
        default=200,
        type=int
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true"
    )
    parser.add_argument(
        "--cosine",
        action="store_true"
    )
    parser.add_argument(
        "--random_seed",
        default=100,
        type=int
    )
    
    #Object selection
    parser.add_argument(
        "--sampling",
        action="store_true"
    )
    parser.add_argument(
        "--sample_num",
        default=1000,
        type=int
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------------------
    #Setting 
    os.makedirs(args.output_dir, exist_ok=True)
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'COCO20k':
        root_path = './datasets/COCO/images/train2014'
    elif args.dataset == 'VOC07':
        root_path = './datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages'
    elif args.dataset == 'VOC12':
        root_path = './datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages'

    if 'base' in args.arch:
        n_head = 12
    else:
        n_head = 6    
    
    print(f'Spherical Kmeans : {args.cosine}')
    
    #Token loading    
    print("Loading Token to Memory")
    token = np.load(args.token_path)
    ####Remove-------------------------------------------------------------------------------------------------------
    ####Remove-------------------------------------------------------------------------------------------------------
    ####Remove-------------------------------------------------------------------------------------------------------
    ####Remove-------------------------------------------------------------------------------------------------------
    ####Remove-------------------------------------------------------------------------------------------------------
    ####Remove-------------------------------------------------------------------------------------------------------
    if args.token_ratio == 1.0:
        pass 
    else:
        print(f"Token Ratio : {args.token_ratio}")
        token = token[:int(token.shape[0] * args.token_ratio) , :]
    #-------------------------------------------------------------------------------------------------------
    print("Token Loading Done")

    #Dimension
    d = token.shape[-1]
    #random seed set
    random_seed = args.random_seed
    
    if args.resize:
        resize = (640,960)
    else:
        resize = None

    #Backbone/Dataloader 
    backbone = get_model(arch=args.arch,
                         patch_size=args.patch_size,
                         device=device)
    dataset, dataloader = dataset_load(dataset_name=args.dataset,
                                       dataset_set=args.set,
                                       resize=resize,
                                       remove_hards=args.remove_hard,
                                       patch_size=args.patch_size)
    
    # -------------------------------------------------------------------------------------------------------
    #Bisecting Kmeans 
    centroid_list = []
    object_idx_list = []
    for n_cluster in range(2,args.n_cluster+1):
        print(f'{n_cluster} Kmeans train....')
        kmeans = faiss.Kmeans(d=d,
                              k=2,
                              niter = args.n_iter,
                              spherical = args.cosine,
                              verbose=True,
                              max_points_per_centroid = 100000000,
                              gpu = args.use_gpu,
                              seed = args.random_seed)
        kmeans.train(token)
        centroid = kmeans.centroids
        print(centroid)
        centroid_list.append(centroid)
    # -------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------------------------
        #Select Obj Cluster
        #Sampling 
        print(f"Object Selection {n_cluster}")
        if args.sampling:
            samples = random.sample(range(len(dataloader)), args.sample_num)
        else:
            samples = range(len(dataloader))
        
        num_one = 0
        num_zero = 0
        for image_num in tqdm(samples):
            padded_img, dims, _, _, _, _, _, _  =get_image(image_num=image_num,
                                                           root_path=root_path,
                                                           dataset=dataset,
                                                           dataloader=dataloader,
                                                           patch_size=args.patch_size,
                                                           device=device,
                                                           return_id=False)
            #Register hool
            feat_out = {}
            def hook_forward(module, inpout, output):
                feat_out['qkv']  = output
            def hook_forward_feature(module, intput, output):
                feat_out['feat'] = output[0][1: , :]
            backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_forward)
            backbone._modules["blocks"][-1].register_forward_hook(hook_forward_feature)

            _ = backbone(padded_img[None, : , : , :].to(device))

            feat = feat_out['feat'].cpu().numpy()
            num_token = feat.shape[0]
            qkv = feat_out['qkv'].reshape(1, num_token + 1, 3, n_head, -1 // n_head).permute(2,0,3,1,4)
            q,k,_ = qkv[0], qkv[1], qkv[2]
            q = q.transpose(1,2).reshape(1, num_token+1, -1)
            k = k.transpose(1,2).reshape(1, num_token+1, -1)

            cls = q[0, 0, :]
            q = q[0, 1: , :]
            k = k[0, 1:, :]

            cls_sim = k@cls
            attn_score = torch.sum((q@k.transpose(1,0))[torch.where(cls_sim > 0)[0]], dim=0)
            attn_score = torch.where(attn_score > 0 , 1 , 0).cpu().numpy()

            for idx, centroid in enumerate(centroid_list):
                kmean = faiss.IndexFlatL2(d)
                kmean.add(centroid)
                _, I = kmean.search(feat, 1)
                I = I.flatten()
                if idx == len(centroid_list) - 1:
                    num_one += np.sum(I[np.where(attn_score > 0 )[0]])
                    num_zero += len(I[np.where(attn_score > 0 )[0]]) - num_one
                    break
                obj_idx = np.where( I == object_idx_list[idx])[0]
                #분할되지 않았으니까 feat == None이 되서 에러 발생 
                if len(obj_idx) == 0: 
                    break 
                feat = feat[obj_idx]
                attn_score = attn_score[obj_idx]

        if num_one > num_zero:
            object_idx_list.append(1) 
            print(f'Selected Object Index | 1')
        else:
            object_idx_list.append(0)
            print(f'Selected Object Index | 0')
            
        # -------------------------------------------------------------------------------------------------------
        kmeans = faiss.IndexFlatL2(d)
        kmeans.add(centroid)
        _, I = kmeans.search(token, 1)
        token = token[np.where(I.flatten() == object_idx_list[-1] )[0] , : ]

    
    centroid_list = np.asarray(centroid_list)
    object_idx_list = np.asarray(object_idx_list)
    if args.cosine:
        np.save(f'{args.output_dir}/{args.dataset}_{args.arch}_{args.patch_size}_{args.n_cluster}_{args.token_ratio}_centroids.npy' , centroid_list)
        np.save(f'{args.output_dir}/{args.dataset}_{args.arch}_{args.patch_size}_{args.n_cluster}_{args.token_ratio}_obj_list.npy' , object_idx_list)
    else:
        np.save(f'{args.output_dir}/{args.dataset}_{args.arch}_{args.patch_size}_{args.n_cluster}_{args.token_ratio}_euclidean_centroids.npy' , centroid_list)
        np.save(f'{args.output_dir}/{args.dataset}_{args.arch}_{args.patch_size}_{args.n_cluster}_{args.token_ratio}_euclidean_obj_list.npy' , object_idx_list)
    # -------------------------------------------------------------------------------------------------------

        


            




    