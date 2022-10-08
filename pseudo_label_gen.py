import torch
import os 
import json
from sklearn.cluster import KMeans, BisectingKMeans
from einops import einops
import faiss
import cv2 
import random 
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from monet_utils import *
from monet_object_discovery import *
from TokenCut.datasets import * 
import matplotlib.pyplot as plt 
from scipy import ndimage
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pseudo Label Gen")
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
        choices=["COCO20k", "VOC12", "VOC07", "COCOval", "COCOtrain"]
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
        default="./ca_pseudo_label",
    )

#Bisectin Kmeans Param
    parser.add_argument(
        "--bisecting_num",
        default=2,
        type=int
    )
    parser.add_argument(
        "--centroid_path",
        default='/data_ssd/monet/BiKmeans_centroid/COCO20k_dino_base_8_10_centroids.npy',
        type=str
    )
    parser.add_argument(
        "--obj_idx_path",
        default='/data_ssd/monet/BiKmeans_centroid/COCO20k_dino_base_8_10_obj_list.npy' ,
        type=str
    )

#Kmeans Refine Param   
    parser.add_argument(
        "--min_cluster",
        default=2,
        type=int
    )
    parser.add_argument(
        "--max_cluster",
        default=20,
        type=int
    )
    parser.add_argument(
        "--region_thred",
        default=0.8,
        type=float
    )
    parser.add_argument(
        "--object_thred",
        default=0.5,
        type=float
    )
    parser.add_argument(
        "--n_iter",
        default=20,
        type=int
    )
    parser.add_argument(
        "--min_box_size",
        default=10,
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
        "--select_type",
        default='cls_sim',
        type=str
    )
    parser.add_argument(
        "--sampling",
        action="store_true"
    )
    parser.add_argument(
        "--sample_num",
        default=5000,
        type=int
    )
    parser.add_argument(
        "--eval",
        action="store_true"
    )
    parser.add_argument(
        "--gt_file_path",
        default='./datasets/instances_train2014_sel20k.json',
        type=str
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------------------
    #Setting 
    os.makedirs(args.output_dir, exist_ok=True)
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'COCO20k':
        root_path = './datasets/COCO/images/train2014'
    elif args.dataset == 'COCOval':
        root_path = './datasets/COCOval/val2017'
    elif args.dataset == 'VOC07':
        root_path = './datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages'
    elif args.dataset == 'VOC12':
        root_path = './datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages'
    elif args.dataset == 'COCOtrain':
        root_path = '/data_ssd/monet/datasets/COCOtrain'

    if 'base' in args.arch:
        n_head = 12
    else:
        n_head = 6    
    
    
    centroids = np.load(args.centroid_path)
    obj_idx = np.load(args.obj_idx_path)

    #Dimension
    d = centroids[0].shape[-1]
    #random seed set
    random_seed = args.random_seed
    
    if args.resize:
        resize = (640, 960)
    else:
        resize = None
    
    print('# -------------------------------------------------------------------------------------------------------')
    print("Paramter Space")
    print(f'Arch : {args.arch} | Patch size : {args.patch_size}')
    print(f'Dataset : {args.dataset}')
    print(f'Region thred : {args.region_thred} | Object thred : {args.object_thred}')
    print('# -------------------------------------------------------------------------------------------------------')


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

    if args.sampling: #for quick eval result not for pseudo label gen 
        pbar = tqdm(range(args.sample_num))
    else:
        pbar = tqdm(range(len(dataloader)))
    result = []
    id_list = []
    clustering_dict = {}
    n_head = 12
    no_anns = 0 
    with torch.no_grad():
        for i in pbar:

            image_num = i
            try:
                padded_img, dims, gt_bbxs, image, vis_image, init_image_size, loader_img_size, filename, image_id = get_image(image_num=image_num,
                                            root_path=root_path,
                                            dataset = dataset,
                                            dataloader=dataloader,
                                            patch_size=args.patch_size,
                                            device=device,
                                            return_id = True)
            except:
                no_anns += 1 
                pass 


            #Register hool
            feat_out = {}
            def hook_forward(module, inpout, output):
                feat_out['qkv']  = output
            def hook_forward_feature(module, intput, output):
                feat_out['feat'] = output[0][1: , :]

            backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_forward)
            backbone._modules["blocks"][-1].register_forward_hook(hook_forward_feature)
            
            _ = backbone(padded_img[None, : , : , :].to(device))

            feature = feat_out['feat']
            num_token = feature.shape[0]
            qkv = feat_out['qkv'].reshape(1, num_token + 1, 3, n_head, -1 // n_head).permute(2,0,3,1,4)
            q,k,v = qkv[0], qkv[1], qkv[2]
            q = q.transpose(1,2).reshape(1, num_token+1, -1)
            k = k.transpose(1,2).reshape(1, num_token+1, -1)
            v = v.transpose(1,2).reshape(1, num_token+1, -1)

            cls = q[0, 0, :]
            query = q[0, 1: , :]
            key = k[0, 1:, :]


            #Bisecting Kmeans 
            max_k = args.bisecting_num
            clst = 1
            object_where = None
            feat = feature.cpu().numpy().copy()
            mask = np.zeros(feature.shape[0])
            for cent, idx in zip(centroids[:max_k], obj_idx[:max_k]):
                kmean = faiss.IndexFlatL2(feature.shape[-1])
                kmean.add(cent)
                _, I = kmean.search(feat, 1)
                if object_where is None:
                    object_where = np.where(I.flatten() == idx)[0]
                else:
                    object_where = object_where[np.where(I.flatten() == idx)[0]]
                mask[object_where] = clst 
                feat = feat[np.where(I.flatten() == idx)[0]]
                clst += 1 

            cls_sim = key@cls
            selected_clst = mask[torch.sort(cls_sim, descending=True)[1][:1].cpu().numpy()]
            mask = np.where(mask == selected_clst[0], 1, 0)
            

            #Kmeans Refine 
            refine_result, refine_thred_result = kmeans_refine(feature=feature,
                                                            mask = mask,
                                                            dims=dims, min_cluster = args.min_cluster, 
                                                            max_cluster = args.max_cluster, max_iter=args.n_iter,
                                                            random_state = args.random_seed,
                                                            ratio=args.region_thred, thred=args.object_thred, cosine=args.cosine, gpu=args.use_gpu,
                                                            return_all=True, 
                                                            )


            seg_mask = object_selec(mask = refine_thred_result, 
                                    cls=cls,
                                    query=query,
                                    key=key,
                                    dims=dims,
                                    obj_select_type = args.select_type, 
                                    top_k=None,
                                    min_box_size=args.min_box_size)



            #Resize to loader image size 
            width, height = padded_img.shape[-1], padded_img.shape[-2]
            ccmask = cv2.resize(seg_mask.reshape(dims), (width, height), interpolation=cv2.INTER_NEAREST)
            
            #Padding된 부분 제거
            #이거 안해주면 mAP 75 이상에서 엄청 떨어짐
            ccmask = ccmask[:loader_img_size[-2] , :loader_img_size[-1]]
            #Zero size
            ccmask[-1, : ] = 0
            ccmask[:, -1] = 0
            width, height = init_image_size[1], init_image_size[0]
            ccmask = cv2.resize(ccmask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            
            record = ccmask_to_coco_ca(ccmask, image_id)
            result.extend(record)
            id_list.append(image_id)
            
            #For Object Clustering 
            bboxs = ccmask_to_box_pred(ccmask)
            segs = ccmask_to_seg_pred(ccmask)
            clustering_dict[image_id] = {
                                        'fileanme' : filename,
                                        'bbox' : bboxs,
                                        'seg' : segs,
                                        'category_id' : [1 for _ in range(len(bboxs))],
                                        'score' : [1.0 for _ in range(len(bboxs)) ]
                                        }
    pseudo_label_filename = f'ca_pseudo_label_{args.dataset}_{args.arch}_{args.patch_size}_{args.bisecting_num}_{args.region_thred}_{args.object_thred}_{args.min_box_size}_{args.select_type}.json'
    pseudo_label_eval_filename = f'ca_pseudo_label_{args.dataset}_{args.arch}_{args.patch_size}_{args.bisecting_num}_{args.region_thred}_{args.object_thred}_{args.min_box_size}_{args.select_type}_eval.json' 
    pseudo_label_path = os.path.join(args.output_dir, pseudo_label_filename)
    pseudo_label_eval_path = os.path.join(args.output_dir, pseudo_label_eval_filename)
    with open(pseudo_label_path, 'w') as f:
        json.dump(clustering_dict, f)
    with open(pseudo_label_eval_path, 'w') as f:
        json.dump(result, f)

    print(f"No annotation file num : {no_anns}")
        
    if args.eval:
        annfile = args.gt_file_path
        cocoGT = COCO(annfile)
        cocoDt = cocoGT.loadRes(pseudo_label_eval_path)
        cocoEval = COCOeval(cocoGT, cocoDt ,'segm' )
        cocoEval.params.imgIds = id_list
        cocoEval.params.useCats = 0 #Class Agnostic

        cocoEval.evaluate()
        cocoEval.accumulate()
        print(cocoEval.summarize())

        stats = cocoEval.stats
        stats_filename =f'ca_cocoeval_{args.dataset}_{args.arch}_{args.patch_size}_{args.bisecting_num}_{args.region_thred}_{args.object_thred}_{args.min_box_size}_{args.select_type}.npy'
        stats_path = os.path.join(args.output_dir+'/stats')
        os.makedirs(stats_path, exist_ok=True)
        stats_path = os.path.join(stats_path, stats_filename)
        np.save(stats_path, stats)

        
