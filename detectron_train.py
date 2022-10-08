
import torch
import numpy as np 
import cv2, os, json, random
import matplotlib.pyplot as plt
import skimage.io as io
from pycocotools.coco import COCO
import pycocotools
import pycocotools.mask as cocomask


import detectron2
from detectron2 import model_zoo
from detectron2.structures import Instances
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import ColorMode
setup_logger()

import os

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)


coco20k_json_file = '/data_ssd/monet/datasets/instances_train2014_sel20k.json'
coco20k_root_path = '/data_ssd/monet/datasets/COCO/images/train2014'
coco20k_pseudo_file = '/data_ssd/monet/train_label/ca_pseudo_label_COCO20k_dino_base_8_2_0.8_0.5_30_all.json'


def coco20k_pseudo_dict(json_file, root_dir = coco20k_root_path):
    dataset_dict = []
    with open(json_file, 'r') as f:
        ann_json = json.load(f)
    for image_id in ann_json.keys():
        image_ann = ann_json[image_id]
        record = {}
        record['file_name'] = os.path.join(root_dir, os.path.basename(image_ann['fileanme']))
        record['image_id'] = image_id
        
        annotations = []
        for i in range(len(image_ann['bbox'])):
            ann_record = {}
            ann_record['iscrowd'] = 0 
            ann_record['category_id'] = 0
            ann_record['bbox'] = image_ann['bbox'][i]
            ann_record['bbox_mode'] = BoxMode.XYXY_ABS
            ann_record['segmentation'] = image_ann['seg'][i]
            if i == 0:
                size = image_ann['seg'][i]['size']
                record['height'] = size[0]
                record['width'] = size[1]
            annotations.append(ann_record)
        record['annotations'] = annotations
        dataset_dict.append(record)
    return dataset_dict

def coco20k_ca_dict(json_file, root_dir=coco20k_root_path):
    coco = COCO(json_file)
    dataset_dict = []
    with open(json_file, 'r') as f:
        ann_json = json.load(f)
    for image in ann_json['images']:
        record = {}
        record['file_name'] = os.path.join(root_dir, image['file_name'])
        record['image_id'] = image['id']
        record['height'] = image['height']
        record['width'] = image['width']
        
        annotations = []
        for ann_id in coco.getAnnIds(image['id']):
            anno = coco.loadAnns(ann_id)[0]
            ann_record = {}
            ann_record['iscrowd'] = anno['iscrowd']
            ann_record['bbox'] = anno['bbox']
            ann_record['bbox_mode'] = BoxMode.XYWH_ABS
            ann_record['category_id'] = 0
            seg = coco.annToRLE(anno)
            ann_record['segmentation'] = {'size' : seg['size'],
                                          'counts' : seg['counts'].decode('utf8')}
            annotations.append(ann_record)
        record['annotations'] = annotations
        dataset_dict.append(record)
        
    return dataset_dict
            
def voc12_seg(ann_file):
    with open(ann_file, 'r') as f:
        ann_ = json.load(f)
    return ann_

def coco20k_pseudo(ann_file):
    with open(ann_file, 'r') as f:
        ann_ = json.load(f)
    return ann_


#CLear 
DatasetCatalog.clear()
MetadataCatalog.clear()

for d in [coco20k_pseudo_file]:
    DatasetCatalog.register('coco20k_pseudo', lambda d = d : coco20k_pseudo_dict(d) )
MetadataCatalog.get("coco20k_pseudo").thing_classes = ["object"]
MetadataCatalog.get("coco20k_pseudo").evaluator_type = "coco"

meta = MetadataCatalog.get("coco20k_pseudo")
pseudo_dicts = coco20k_pseudo_dict(coco20k_pseudo_file)

for d in [coco20k_json_file]:
    DatasetCatalog.register('coco20k_ca', lambda d = d : coco20k_ca_dict(d) )
MetadataCatalog.get("coco20k_ca").thing_classes = ["object"]
meta = MetadataCatalog.get("coco20k_ca")
dataset_dicts = coco20k_ca_dict(coco20k_json_file)


"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""


from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels



def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["coco"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )