import torch
import argparse
from victim_detector.models.yolo import Model
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import torchvision.transforms as TF
import kornia.geometry.transform as KT
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm.auto import tqdm
from utils.plots import output_to_target, plot_images, plot_val_study
from threading import Thread

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from victim_detector.models.common import *
from victim_detector.models.experimental import *
from victim_detector.utils.autoanchor import check_anchor_order
from victim_detector.utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from victim_detector.utils.plots import feature_visualization
from victim_detector.utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
from victim_detector.utils.loss import ComputeLoss, MyComputeLoss
from victim_detector.utils.dataloaders import create_dataloader
from victim_detector.utils.autobatch import check_train_batch_size
from victim_detector.utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from victim_detector.utils.metrics import fitness

from utils_patch import *
import val_patch
from pso import OptimizeFunction, PSO

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

SEED = 42
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    parser.add_argument('--data', type=str, default='victim_detector/data/custom.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='runs/aug/weights/best.pt', help='model.pt path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    
    parser.add_argument('--patch_size', type=float, default=0.14)
    
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    hyp = 'victim_detector/data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    
    weights = opt.weights
    save_dir = Path("/".join(weights.split("/")[:-2]))

    print(os.getcwd(), weights)
    ckpt = torch.load(weights, map_location='cpu')
    # print(ckpt['model'].yaml)
    # raise NotImplementedError
    # model = Model(ckpt['model'].yaml, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)
    model = ckpt['model'].to(device)
    # print(model)
    # model_alt = torch.load('/home/lukas/PycharmProjects/HotCold/HOTCOLDBlock/assets/yolov5s.pt')
    # exclude = ['anchor'] if (hyp.get('anchors')) and not resume else []
    # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
    # model.load_state_dict(csd, strict=False)
    
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # Batch size
    batch_size = opt.batch_size
    
    # Dataloader
    print(os.getcwd(), opt.data)
    data_dict = check_dataset(opt.data)
    print(data_dict)
    train_path, val_path = data_dict['train'], data_dict['val']
    train_loader, dataset = create_dataloader(train_path,
                                              opt.imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              opt.single_cls,
                                              hyp=hyp,
                                              augment=False,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=opt.workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    val_loader = create_dataloader(val_path,
                                   opt.imgsz,
                                   batch_size // WORLD_SIZE * 2,
                                   gs,
                                   opt.single_cls,
                                   hyp=hyp,
                                   cache=None if opt.noval else opt.cache,
                                   rect=True,
                                   rank=-1,
                                   workers=opt.workers * 2,
                                   pad=0.,
                                   prefix=colorstr('val: '))[0]
    
    model.eval()
    
    nb = len(train_loader)
    
    # PSO
    pso = PSO(100, device)
    func = OptimizeFunction(model, opt.patch_size, device)
    
    # save_dir = Path('results')
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    best_fitness = 100.0

    for epoch in range(opt.epochs):  # epoch ------------------------------------------------------------------
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            targets = targets.to(device)
            imgs = imgs.to(device, non_blocking=True).float() / 255      # (n, c, h, w)
            
            func.set_para(targets, imgs)
            pso.optimize(func)
            swarm_parameters = pso.run()

            # break
            
        # val
        results, maps, _ = val_patch.run(data_dict,
                                         patch=swarm_parameters,
                                         opt=opt,
                                         batch_size=batch_size // WORLD_SIZE * 2,
                                         imgsz=imgsz,
                                         model=model,
                                         single_cls=opt.single_cls,
                                         dataloader=val_loader,
                                         plots=False,
                                         save_dir=save_dir
                                         )
        with open(f"{save_dir}/res_hcb.txt", "a") as f:
            f.write(f'gbest_position: {swarm_parameters.gbest_position[0]}\n')
            f.write(f'{swarm_parameters.gbest_position[1]}')
            f.write(f'gbest_value: {swarm_parameters.gbest_value}\n\n')

