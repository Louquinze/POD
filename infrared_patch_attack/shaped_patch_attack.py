import os
import io
import joblib
import time
import argparse
from PIL import Image
from config import Config
from torchvision import datasets, transforms
from attack.mask_attack import shaped_mask_attack_2
from yolov5.detect import load_model, detect
from utils.dataloaders import create_dataloader
from utils.general import non_max_suppression

from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os.path as osp
import json

import torch
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from yolov7.detect_attack import load_model, detect
# from yolov5.detect_attack import load_model, detect

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

loader = transforms.Compose([
    transforms.ToTensor()
])
conf_thre = 0.4


def eval_coco_metrics(anno_json: str, pred_json: str, txt_save_path: str, w_mode: str = 'a') -> np.ndarray:
    """
    Compare and eval pred json producing coco metrics
    """
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    evaluator = COCOeval(anno, pred, 'bbox')

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # capture evaluator stats and save to file
    std_out = io.StringIO()
    with redirect_stdout(std_out):
        evaluator.summarize()
    eval_stats = std_out.getvalue()
    return evaluator.stats


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def attack_process(H, W, img, threat_model, device, emp_iterations, max_pertubation_mask, content, folder_path, name,
                   grad_avg, bbox):
    # input = loader(img)
    # input = input.repeat((3, 1, 1))
    input = img[0]
    # print(input.shape)
    # bbox, prob, _ = detect(threat_model, input) # 在攻击前检测原目标的置信度
    begin = time.time()
    adv_img_ts, adv_img, mask = shaped_mask_attack_2(H, W, bbox, threat_model, img, device, emp_iterations,
                                                     max_pertubation_mask, content, grad_avg)  # 调用攻击函数进行攻击
    _, prob, _ = detect(threat_model, adv_img_ts)  # 在攻击后检测目标的置信度
    end = time.time()
    print("optimization time: {}".format(end - begin))
    print("obj score after attack: ", prob)

    imgs_dir = os.path.join(folder_path, "adv_imgs").replace("\\", "/")
    msks_dir = os.path.join(folder_path, "infrared_masks").replace("\\", "/")

    file_name = name.replace("\\", "/").split("/")[-1]
    img_path = os.path.join(imgs_dir, file_name)
    msk_path = os.path.join(msks_dir, file_name)

    adv_img.save(img_path, quality=99)
    mask.save(msk_path, quality=99)

    if prob < conf_thre:
        return True, adv_img_ts, prob
        # joblib.dump(adv_img_ts,folder_path+"/res/adv_ts" + str(args.number) + "+" + str(args.max_pertubation_mask) + "/{}_pedestrian&params.pkl".format(name))
        # joblib.dump(mask,folder_path+"/res/mask" + str(args.number) + "+" + str(args.max_pertubation_mask) + "/{}_mask&params.pkl".format(name))  
    else:
        return False, adv_img_ts, prob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--victim_imgs', type=str, default="", help='the folder with victim images to conduct attacks')
    parser.add_argument('--model', type=str, default="patch_det_1",
                        # /home/lukas/PycharmProjects/AttackExp/runs/train/patch_det_1/weights/best.pt
                        help='the folder with victim images to conduct attacks')
    args = parser.parse_args()
    ## 加载攻击参数 ##
    opt = Config()
    save_folder = f"res/{args.model}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    imgs_dir = os.path.join(save_folder, "adv_imgs").replace("\\", "/")
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)
    msks_dir = os.path.join(save_folder, "infrared_masks").replace("\\", "/")
    if not os.path.exists(msks_dir):
        os.mkdir(msks_dir)

    ## 加载待攻击模型 ##
    threat_model = load_model(
        f"../runs/train/{args.model}/weights/last.pt")
    threat_model.eval()
    threat_model = threat_model.to("cuda:0")

    gs = max(int(threat_model.stride), 32)  # grid size (max stride)
    single_cls = True  # Todo add args

    # Trainloader
    train_loader, dataset = create_dataloader(
        "../../datasets/CVC-Sub/valid/images",
        # "../../datasets/CVC-Sub/valid/images",
        416,
        1,
        gs,
        single_cls,
        # hyp=hyp,
        augment=False,
        # apply_patches=opt.apply_patch,
        # apply_adv=opt.apply_adv,
        # detect_patches=opt.detect_patch,
        # path_adv=adv_dir,
        # cache=None if opt.cache == 'val' else opt.cache,
        #  rect=opt.rect,
        rank=-1,
        workers=4,
        # image_weights=opt.image_weights,
        # quad=opt.quad,
        prefix='train: ',
        shuffle=False,
        seed=42,
        # patch_dir=opt.patch_dir
    )
    # for i,name in enumerate(os.listdir(folder_path)):
    #     file_path = os.path.join(folder_path, name)
    """with open(os.path.join(folder_path, name).replace("images", "labels").replace("jpg", "txt"), "r") as f:
        lab = f.read()

    stacked_tensors = []
    for string in lab.split("\n")[:-1]:
        numbers = [float(num) for num in string.split()]
        tensor = torch.tensor(numbers)[1:]
        stacked_tensors.append(tensor)
        # conf_thres=0.01 # confidence threshold
    pred = torch.stack(stacked_tensors) * 640"""

    # stores json results
    clean_gt_results = []
    clean_results = []
    patch_results = []

    clean_image_annotations = []
    box_id = 0

    for i, (img, targets, paths, _) in enumerate(train_loader):
        clean_image_annotations.append({
            "file_name": paths,
            "height": 416,
            "width": 416,
            "id": i,
        })
        # img = Image.open(file_path)
        # input = loader(img)
        # input = input.repeat((3, 1, 1))
        # bbox, prob, fp = detect(threat_model, img)  # 在攻击前检测原目标的置信度
        # (68, 345, 146, 267) tensor(0.77366, device='cuda:0')
        # print(bbox, prob, fp)
        img = img.type(torch.FloatTensor).to("cuda:0") / 255
        # print(threat_model(img), targets, paths)
        # print(targets[0][2:])
        # print(targets[0][2:] * 416)
        bbox = xywh2xyxy(targets[0][2:])
        bbox = [int(i) for i in bbox * 416]
        # bbox[0], bbox[1] = bbox[1], bbox[0]
        bbox[2], bbox[3], bbox[0], bbox[1] = bbox[0], bbox[2], bbox[1], bbox[3]
        print("{}th image".format(i))
        flag_2 = True
        prob_max = 1.
        for k in range(opt.iterations):
            print("{}th attack".format(k))
            flag, adv_img_t, prob = attack_process(opt.height, opt.width, img, threat_model, opt.device, opt.emp_iterations,
                                           opt.max_pertubation_mask, opt.content, save_folder, paths[0].split("/")[-1],
                                           opt.grad_avg, bbox=bbox)
            if prob < prob_max:
                adv_img = adv_img_t
                prob_max = prob

        labels = []
        pred = threat_model(img)
        boxes = non_max_suppression(pred, 0.5, 0.5)[0]
        for box in boxes:
            cls_id_box = box[-1].item()
            score = box[4].item()
            x_center, y_center, width, height = box[:4]
            x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
            labels.append(
                [cls_id_box, x_center / 416, y_center / 416, width / 416, height / 416])
            clean_results.append(
                {'image_id': i,
                 'bbox': [x_center - width / 2, y_center - height / 2, width, height],
                 'score': round(score, 5),
                 'category_id': int(cls_id_box)})
            clean_gt_results.append(
                {'id': box_id,
                 "iscrowd": 0,
                 'image_id': i,
                 'bbox': [x_center - width / 2, y_center - height / 2, width, height],
                 'area': width * height,
                 'category_id': int(cls_id_box),
                 "segmentation": []})
            box_id += 1

        labels = []
        pred = threat_model(adv_img.to("cuda:0"))
        boxes = non_max_suppression(pred, 0.5, 0.5)[0]
        for box in boxes:
            cls_id_box = box[-1].item()
            score = box[4].item()
            x_center, y_center, width, height = box[:4]
            x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
            labels.append(
                [cls_id_box, x_center / 416, y_center / 416, width / 416, height / 416])
            patch_results.append(
                {'image_id': i,
                 'bbox': [x_center - width / 2, y_center - height / 2, width, height],
                 'score': round(score, 5),
                 'category_id': int(cls_id_box)})
            box_id += 1

    # add all required fields for a reference GT clean annotation
    clean_gt_results_json = {"annotations": clean_gt_results,
                             "categories": [],
                             "images": clean_image_annotations}
    for index, label in enumerate(["person"], start=0):
        categories = {"supercategory": "Defect",
                      "id": index,
                      "name": label}
        clean_gt_results_json["categories"].append(categories)

    # save all json results

    print(clean_results)
    print(clean_gt_results_json)

    clean_gt_json = osp.join(save_folder, 'clean_gt_results.json')
    clean_json = osp.join(save_folder, 'clean_results.json')
    patch_json = osp.join(save_folder, 'patch_results.json')

    with open(clean_gt_json, 'w', encoding="utf-8") as f_json:
        json.dump(clean_gt_results_json, f_json, ensure_ascii=False, indent=4)
    with open(clean_json, 'w', encoding="utf-8") as f_json:
        json.dump(clean_results, f_json, ensure_ascii=False, indent=4)
    with open(patch_json, 'w', encoding="utf-8") as f_json:
        json.dump(patch_results, f_json, ensure_ascii=False, indent=4)

    res_c = eval_coco_metrics(clean_gt_json, clean_json, "clean_test.txt")
    res_p = eval_coco_metrics(clean_gt_json, patch_json, "patch_test.txt")
    with open(f"{save_folder}/res_clean.txt", "w") as res_file:
        res_file.write(str(res_c))
    with open(f"{save_folder}/res_patch.txt", "w") as res_file:
        res_file.write(str(res_p))
