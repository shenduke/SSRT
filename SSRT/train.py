# flake8: noqa
import os.path as osp

import ssrt.archs
import ssrt.data
import ssrt.models
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    #print(root_path)
    train_pipeline(root_path)
