import os
import joblib
import time
import argparse
from PIL import Image
from config import Config
from torchvision import datasets, transforms
from attack.mask_attack import shaped_mask_attack_1
from yolov5.detect import load_model, detect
# from yolov7.detect_attack import load_model, detect
# from yolov5.detect_attack import load_model, detect

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

loader = transforms.Compose([
    transforms.ToTensor()
])
conf_thre = 0.5

def attack_process(H, W, img, threat_model, device, emp_iterations, max_pertubation_mask, content, folder_path, name, grad_avg):
    input = loader(img)
    # input = input.repeat((3, 1, 1))
    print(input.shape)
    bbox, prob, _ = detect(threat_model, input) # 在攻击前检测原目标的置信度
    begin = time.time()
    adv_img_ts, adv_img, mask = shaped_mask_attack_1(H, W, bbox, threat_model, img, device, emp_iterations, max_pertubation_mask, content, grad_avg) # 调用攻击函数进行攻击
    _, prob, _ = detect(threat_model,adv_img_ts) # 在攻击后检测目标的置信度
    end = time.time()
    print("optimization time: {}".format(end - begin))
    print("obj score after attack: ",prob)
    if prob < conf_thre:
        imgs_dir = os.path.join(folder_path, "adv_imgs")
        msks_dir = os.path.join(folder_path, "infrared_masks")
        img_path = os.path.join(imgs_dir, name)
        adv_img.save(img_path,quality=99)
        msk_path = os.path.join(msks_dir, name)
        mask.save(msk_path,quality=99)
        return True
        # joblib.dump(adv_img_ts,folder_path+"/res/adv_ts" + str(args.number) + "+" + str(args.max_pertubation_mask) + "/{}_pedestrian&params.pkl".format(name))
        # joblib.dump(mask,folder_path+"/res/mask" + str(args.number) + "+" + str(args.max_pertubation_mask) + "/{}_mask&params.pkl".format(name))  
    else: return False


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--victim_imgs',type=str,default="", help='the folder with victim images to conduct attacks')
    parser.add_argument('--model',type=str,default="base_train1", help='the folder with victim images to conduct attacks')
    args = parser.parse_args()
    ## 加载攻击参数 ##
    opt = Config()
    if args.victim_imgs == "":
        folder_path = opt.attack_dataset_folder
        if not os.path.join(folder_path):
            print("please prepare the dataset correctly")
            assert False
    else: folder_path = args.victim_imgs
    save_folder = f"res/{args.model}"
    print(save_folder, folder_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    imgs_dir = os.path.join(save_folder, "adv_imgs")
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)
    msks_dir = os.path.join(save_folder, "infrared_masks")
    if not os.path.exists(msks_dir):
        os.mkdir(msks_dir)
    
    ## 加载待攻击模型 ##
    threat_model = load_model(f"/home/fmg/v-strack/yolov5_adversarial/runs/train/{args.model}/weights/best.pt")
    threat_model.eval()

    suc = 0
    sum = 0
    tp = 0
    fp = 0
    fn = 0
    for i,name in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, name)
        img = Image.open(file_path)
        input = loader(img)
        # input = input.repeat((3, 1, 1))
        bbox, prob, fp = detect(threat_model, input) # 在攻击前检测原目标的置信度
        print("{}th image".format(i))
        if prob<0.5: # 本身检测分数较低的跳过
            print("detector cannot detect any pedestrian in the image")
            fn += 1
            continue
        else:
            print("obj score before attack: ", prob)
        sum += 1
        flag_2 = True
        for k in range(opt.iterations):
            print("{}th attack".format(k))
            flag = attack_process(opt.height, opt.width, img, threat_model, opt.device, opt.emp_iterations, opt.max_pertubation_mask, opt.content, save_folder, name, opt.grad_avg)
            if flag:
                fn += 1
                suc += 1
                flag_2 = False
                break

        if flag_2:
            tp += 1

        try:
            p = tp / (tp + fp)
            recall = tp / (tp + fn)
            print(f"Precision: {p}")
            print(f"Recall: {recall}")
        except:
            pass
        print(f"ASR: {float(suc) / sum}\n")

    recall = tp / (tp + fn)
    print("ASR: ", float(suc)/sum)
    with open(f"{save_folder}/res.txt", "a") as f:
        f.write(f"ASR: {float(suc)/sum}\n")
        if tp > 0 or fp > 0:
            p = tp / (tp + fp)
            f.write(f"Precision: {p}\n")
        f.write(f"Recall: {recall}\n\n")

           