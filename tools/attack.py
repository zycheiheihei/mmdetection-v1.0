from __future__ import division
import argparse
import os

import torch
from mmcv import Config
from mmdet import __version__
from train import parse_args,train


def main():
    args = parse_args()
    train(args)
    model = torch.load(args.work_dir + "/" + args.model_name)
    model.eval()


if __name__ == "__main__":
    main()
