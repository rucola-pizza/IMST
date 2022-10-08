"""
Utils 
Monet Flow 
    1. Model & Dataset Load
        Model - dino small base, ibot small, patch size 8,16
        Dataset - COCO20k, VOC12, VOC07
    2. Get image & model input 
        Gert image from image number 
        image to model input, masking(optional)
    3. Model output - Feature 
        key, query, value, output 
    4. Monet detect & segment pseudo mask 
    5. Eval & Vis 
"""
import cv2
import torch
import faiss
import torch.nn as nn 
import skimage.io as io 
from monet_datasets import *
import dino.vision_transformer as vits
from pycocotools import mask as coco_mask 

def get_model(arch, patch_size, device):
    
    if 'dino' in arch:
        _arch = "vit" + arch[4:]
        model = vits.__dict__[_arch](patch_size=patch_size, num_classes=0)

        for p in model.parameters():
            p.requires_grad = False
        
        #load pre-trained model
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
        
        model.eval()   
        model.to(device)
        
    elif 'ibot' in arch:
        print(f"IBot zz ")
        state_dict = torch.load(f'./{arch}_{patch_size}.pth')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if 'small' in arch:
            model = vit_small(patch_size=patch_size, return_all_tokens=True).to(device)
        elif 'base' in arch:
            model = vit_base(patch_size=patch_size, return_all_tokens=True).to(device)       
        elif 'large' in arch:
            model = vit_large(patch_size=patch_size, return_all_tokens=True).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    elif 'mae' in arch:
        print('MAE')
        return None
    else:
        print('Not proper arch')
        return None
    
    return model
        
def dataset_load(dataset_name,  patch_size, dataset_set='train', remove_hards=True,
                 resize=None,):
    dataset = Dataset(dataset_name=dataset_name, dataset_set=dataset_set, patch_size=patch_size,
                      remove_hards=remove_hards, resize=resize,)
    dataloader = dataset.dataloader
    return dataset, dataloader

def img_processing(img, patch_size, device):
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

def get_image(image_num, root_path, patch_size, dataset, dataloader, device, return_id=False, gt=True):
    inp = dataloader[image_num]
    img = inp[0]
    if type(inp[1]) == list: #COCO
        image_id = inp[1][0]['image_id']
        im_name = dataset.get_image_name(inp[1])
        filename = os.path.join(root_path, im_name)
        image = io.imread(os.path.join(root_path, im_name))
    elif type(inp[1]) == dict: #Pascal
        image_id = inp[1]['annotation']['filename'].split('.')[0]
        im_name = dataset.get_image_name(inp[1])
        filename = os.path.join(root_path, im_name)
        image = io.imread(os.path.join(root_path, im_name))
    else: #Pascal Seg
        gt = False
        filename = inp[1].filename
        image_id = os.path.basename(inp[1].filename).split('.')[0]
        image = io.imread(inp[1].filename)

    init_image_size = image.shape
    
    loaded_image_shape = img.shape
    padded_img, w_featmap, h_featmap = img_processing(img, patch_size, device)
    
    if gt:
        gt_bbxs, _ = dataset.extract_gt(inp[1], im_name)

        if gt_bbxs.shape[0] == 0:
            pass 
        else:
            vis_gt_image = image.copy()
            for gt_bbx in gt_bbxs:
                cv2.rectangle(
                    vis_gt_image,
                    (int(gt_bbx[0]), int(gt_bbx[1])),
                    (int(gt_bbx[2]), int(gt_bbx[3])),
                    (255, 0, 0), 2,
                )
    else:
        vis_gt_image = image.copy()
        gt_bbxs = []

    if return_id:
        return padded_img, [w_featmap, h_featmap], gt_bbxs, image,vis_gt_image, init_image_size, loaded_image_shape, filename, image_id, 
        
    return padded_img, [w_featmap, h_featmap], gt_bbxs, image,vis_gt_image, init_image_size, loaded_image_shape, filename

def get_padded_image(image_num,  dataloader, device, patch_size, return_dims=False ):
    padded_img, w_featmap, h_featmap = img_processing(img= dataloader[image_num][0], patch_size = patch_size, device=device, )
    if return_dims:
        return padded_img, [w_featmap, h_featmap]
    return padded_img
    
def get_feature(padded_img, model, device):
    return model.get_features(padded_img.unsqueeze(0).to(device))

def get_single_feature(image_num, model, depth, which_feature, dataset, dataloader, root_path, device, patch_size,
                       masking=None):
    padded_img, dims, _, _, _, _ = get_image(image_num=image_num,
                                             root_path=root_path,
                                             patch_size=patch_size,
                                             dataset=dataset,
                                             dataloader=dataloader,
                                             device=device)
    if masking is not None:
        padded_img = padded_img * masking

    _, output_list, q_list, k_list, v_list = get_feature(padded_img, model, device)
    if which_feature == 'output':
        feature = output_list[depth][: ,: ,:]
    elif which_feature == 'key':
        feature = k_list[depth][: , 1:, :]
    elif which_feature == 'query':
        feature = q_list[depth][: , 1:, :]
    elif which_feature == 'value':
        feature = v_list[depth][: , 1:, :]
    else:
        print("Choose feature from (key query value output)")
        return None
    return feature, dims, q_list, k_list


def gen_mask(padded_img, feature, dims, model, device, topk, centroid_path, sorted_cluster_path):
    #padded_img, feature -> cuda 
    #Load kmeans 
    centroid = np.load(centroid_path)
    kmeans = faiss.IndexFlatL2(centroid.shape[-1])
    kmeans.add(centroid)
    sorted_cluster = np.load(sorted_cluster_path)
    
    _, I_1 = kmeans.search(feature[0, : ,:].cpu().numpy(), 1)

    def in_cluster(x):
        if x in sorted_cluster[:topk]:
            return 1
        return 0
    vec_in_cluster = np.vectorize(in_cluster)
    I_1 = vec_in_cluster(I_1.flatten())

    seg_mask = torch.from_numpy(I_1.reshape(dims)).unsqueeze(0).unsqueeze(0)
    seg_mask = nn.functional.interpolate(input=seg_mask.type(torch.float),
                                         size = padded_img.shape[1:],
                                         mode = 'nearest')
    masked_img = padded_img * seg_mask.squeeze(0)

    _, feature, _, _, _ = get_feature(masked_img.type(torch.float), model, device)

    feature = feature[-1]
    feature = feature[: , np.where(I_1 == 1)[0] , :]
    
    return feature  
    
    
#List[list, list, ... list]
def ccmask_to_box_pred(ccmask):
    pred_result = []
    for obj in np.unique(ccmask):
        if obj == 0:
            continue
        mask = np.where(ccmask == obj)
        
        ymin, ymax = int(min(mask[0])), int(max(mask[0]) + 1)
        xmin, xmax = int(min(mask[1])), int(max(mask[1]) + 1)
        
        pred = [xmin, ymin, xmax, ymax]
        pred_result.append(pred)
    return pred_result

def ccmask_to_seg_pred(ccmask):
    seg_result = []
    for obj in np.unique(ccmask):
        if obj == 0:
            continue # background
        mask = np.asfortranarray(np.where(ccmask == obj, 1, 0).astype(np.uint8))
        rle = coco_mask.encode(mask)
        rle['counts'] = rle['counts'].decode('utf8')
        seg_result.append(rle)
    return seg_result

def ccmask_to_coco_ca(ccmask, image_id):
    coco_result = []
    pred_result = ccmask_to_box_pred(ccmask)
    seg_result = ccmask_to_seg_pred(ccmask)
    for i in range(len(pred_result)):
        record = {
            'image_id' : image_id,
            'category_id' : 1,
            'segmentation' : seg_result[i], 
            'bboxs' : pred_result[i],
            'score' : 1.0    
        }
        coco_result.append(record)
    return coco_result