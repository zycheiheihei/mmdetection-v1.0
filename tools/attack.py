from __future__ import division
import argparse
import os
from mmdet import __version__
import torch
from mmcv import Config
from parsing import parse_args
from train import *
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.apis import get_root_logger, init_dist, set_random_seed
from mmdet.apis.train import *
import pdb
import os
import numpy as np
from tqdm import trange
from mmdet.models import build_detector
from mmdet.datasets import build_dataset


def load_model(args):
    if args.train:
        train(args)
        model = torch.load(args.work_dir + '/' + args.model_name + '.pth')
    else:
        if args.model_path is None:
            print('Model path missing!')
            return
        model = init_detector(args.config, args.model_path, device='cuda:0')
    model.eval()
    return model

def visualize():
    args = parse_args()
    model = load_model(args)
    root = '/home/fengyao/MSCOCO2017dataset/test/test2017/'
    save_path = '/home/fengyao/MSCOCO2017dataset/test/test2017/output/'
    imgs = ['000000000001.jpg',
            '000000000016.jpg']
    for i in range(0, len(imgs)):
        result = inference_detector(model,root + imgs[i])
        show_result(root + imgs[i], result, model.CLASSES,show=False,out_file=save_path + imgs[i])
    print('[INFO]Done.')


def visualize_img(model, img, save_path):
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, show=False, out_file=save_path)
    return


def attack():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        args.work_dir = cfg.work_dir
    args.save_path = args.work_dir + '/output/'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]
    #if len(cfg.workflow) == 2:
        #datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__, config=cfg.text, CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    attack_detector(args, model, cfg, datasets[0])
    return

if __name__ == "__main__":
    #visualize()
    attack()
