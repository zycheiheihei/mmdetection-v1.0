from __future__ import division
import argparse
import os
from mmdet import __version__
import torch
from mmcv import Config
from train import parse_args,train


def main():
    args = parse_args()
    train(args)
    model = torch.load(args.work_dir + "/" + args.model_name)
    model.eval()


if __name__ == "__main__":
    main()
