"""
Token Gen
    1.Masking이 없으면 output feature 를 추출해서 저장 
    2.Masking이 있으면 masking 적용하고 output feature 와 cls 토큰을 저장 
        미리 저장된 centroid를 불러와서 지정된 조건하에 kmeans를 수행해서 mask 생성 
    
"""
from monet_utils import *
import torch 
from monet_datasets import *


def token_gen(model, dataset, dataloader, root_path, patch_size,
              start_num, end_num,
              depth, device,
              masking=False,
              topk=None,
              centroid_path = None,
              sorted_cluster_path = None):
    batch_token_cpu = None
    for image_num in tqdm(range(start_num, end_num+1)):
        padded_img, dims, _, _, _, _, _, _, _ = get_image(image_num=image_num,
                                                 root_path=root_path,
                                                 patch_size=patch_size,
                                                 dataset=dataset,
                                                 dataloader=dataloader,
                                                 device=device,
                                                 return_id=True
                                                 )
        #Register hool
        feat_out = {}
        def hook_forward_feature(module, intput, output):
            feat_out['feat'] = output[0][1: , :]

        model._modules["blocks"][-1].register_forward_hook(hook_forward_feature)
        _ = model(padded_img[None, : ,: ,:].to(device))
        feature = feat_out['feat']

        if masking:
            feature = gen_mask(padded_img=padded_img,
                            feature=feature,
                                dims=dims,
                                model=model,
                                device=device,
                                topk=topk,
                                centroid_path=centroid_path,
                                sorted_cluster_path = sorted_cluster_path
                                )
        
        if image_num == start_num:
            batch_token_gpu = feature
        elif image_num % 50 == 0:
            if batch_token_cpu is not None:
                batch_token_gpu = batch_token_gpu.detach().to('cpu')
                batch_token_cpu = torch.cat([batch_token_cpu, batch_token_gpu], dim=-2)
                batch_token_gpu = feature
            else:
                batch_token_cpu = batch_token_gpu.detach().to('cpu')
                batch_token_gpu = feature
        else:
            batch_token_gpu = torch.cat([batch_token_gpu, feature], dim=-2)
        
        if image_num == end_num:
            batch_token_cpu = torch.cat([batch_token_cpu, batch_token_gpu.detach().to('cpu')], dim=-2)
    
    torch.cuda.empty_cache()
    del batch_token_gpu

    return batch_token_cpu.squeeze()
    