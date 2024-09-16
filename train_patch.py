"""
Training code for Adversarial patch training

python train_patch.py --cfg config_json_file
"""
import os
import os.path as osp
import time
import json
from contextlib import nullcontext

import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
from torch import optim, autograd
from torch.cuda.amp import autocast
from torchvision import transforms as T

# from tensorboardX import SummaryWriter
# from tensorboard import program

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, xyxy2xywh

from helper.test_patch import PatchTester
from adv_patch_gen.utils.config_parser import get_argparser, load_config_object
from adv_patch_gen.utils.common import is_port_in_use, pad_to_square, IMG_EXTNS
from adv_patch_gen.utils.dataset import YOLODataset
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from adv_patch_gen.utils.loss import MaxProbExtractor, SaliencyLoss, TotalVariationLoss, NPSLoss

# setting benchmark to False reduces training time for our setup
# torch.backends.cudnn.benchmark = False

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class PatchTrainer:
    """
    Module for training on dataset to generate adv patches
    """

    def __init__(self, cfg: edict):
        self.cfg = cfg
        self.dev = select_device(cfg.device)

        model = DetectMultiBackend(cfg.weights_file, device=self.dev, dnn=False, data=None, fp16=False)
        self.model = model.eval()

        self.patch_transformer = PatchTransformer(
            cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_off_loc, cfg.y_off_loc, self.dev).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)
        self.prob_extractor = MaxProbExtractor(cfg).to(self.dev)
        self.sal_loss = SaliencyLoss().to(self.dev)
        self.nps_loss = NPSLoss(cfg.triplet_printfile, cfg.patch_size).to(self.dev)
        self.tv_loss = TotalVariationLoss().to(self.dev)

        # freeze entire detection model
        for param in self.model.parameters():
            param.requires_grad = False

        # set log dir
        """count = 0
        while True:
            isExist = os.path.exists(cfg.log_dir)
            if not isExist:
                break
            count += 1
            cfg.log_dir = cfg.log_dir + str(count)"""
        if not os.path.exists(cfg.log_dir):
            os.makedirs(cfg.log_dir)
        # self.writer = self.init_tensorboard(cfg.log_dir, cfg.tensorboard_port)
        self.writer = None
        # save config parameters to tensorboard logs
        if self.writer is not None:
            for cfg_key, cfg_val in cfg.items():
                self.writer.add_text(cfg_key, str(cfg_val))

        # setting train image augmentations
        transforms = None
        if cfg.augment_image:
            transforms = T.Compose(
                [T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
                T.ColorJitter(brightness=.2, hue=.04, contrast=.1),
                T.RandomAdjustSharpness(sharpness_factor=2)])

        # load training dataset
        self.train_loader = torch.utils.data.DataLoader(
            YOLODataset(image_dir=cfg.image_dir,
                        label_dir=cfg.label_dir,
                        max_labels=cfg.max_labels,
                        model_in_sz=cfg.model_in_sz,
                        use_even_odd_images=cfg.use_even_odd_images,
                        transform=transforms,
                        filter_class_ids=cfg.objective_class_id,
                        min_pixel_area=cfg.min_pixel_area,
                        shuffle=True),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.dev.type == "cuda" else False)
        self.epoch_length = len(self.train_loader)

    def init_tensorboard(self, log_dir: str = None, port: int = 6006, run_tb=True):
        """
        Initialize tensorboard with optional name
        """
        if run_tb:
            while is_port_in_use(port) and port < 65535:
                port += 1
                print(f"Port {port - 1} is currently in use. Switching to {port} for tensorboard logging")

            tboard = program.TensorBoard()
            tboard.configure(argv=[None, "--logdir", log_dir, "--port", str(port)])
            url = tboard.launch()
            print(f"Tensorboard logger started on {url}")

        if log_dir:
            return SummaryWriter(log_dir)
        return SummaryWriter()

    def generate_patch(self, patch_type: str, pil_img_mode: str = "RGB") -> torch.Tensor:
        """
        Generate a random patch as a starting point for optimization.

        Arguments:
            patch_type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
            pil_img_mode: Pillow image modes i.e. RGB, L https://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes
        """
        p_c = 1 if pil_img_mode in {"L"} else 3
        p_w, p_h = self.cfg.patch_size
        if patch_type == 'gray':
            adv_patch_cpu = torch.full((p_c, p_h, p_w), 0.5)
        elif patch_type == 'random':
            adv_patch_cpu = torch.rand((p_c, p_h, p_w))
        return adv_patch_cpu

    def read_image(self, path, pil_img_mode: str = "RGB") -> torch.Tensor:
        """
        Read an input image to be used as a patch

        Arguments:
            path: Path to the image to be read.
        """
        patch_img = Image.open(path).convert(pil_img_mode)
        patch_img = T.Resize(self.cfg.patch_size)(patch_img)
        adv_patch_cpu = T.ToTensor()(patch_img)
        return adv_patch_cpu

    def train(self) -> None:
        """
        Optimize a patch to generate an adversarial example.
        """
        # make output dirs
        count = 0
        for elem in os.listdir(self.cfg.log_dir):
            if "patches" in elem:
                count += 1
        patch_dir = osp.join(self.cfg.log_dir, f"patches{count}")

        os.makedirs(patch_dir, exist_ok=True)
        if self.cfg.debug_mode:
            for img_dir in ["train_patch_applied_imgs", "val_clean_imgs", "val_patch_applied_imgs"]:
                os.makedirs(osp.join(self.cfg.log_dir, img_dir), exist_ok=True)

        # dump cfg json file
        # with open(osp.join(self.cfg.log_dir, "cfg.json"), 'w', encoding='utf-8') as json_f:
        #     json.dump(self.cfg, json_f, ensure_ascii=False, indent=4)

        # fix loss targets
        loss_target = self.cfg.loss_target
        if loss_target == "obj":
            self.cfg.loss_target = lambda obj, cls: obj
        elif loss_target == "cls":
            self.cfg.loss_target = lambda obj, cls: cls
        elif loss_target in {"obj * cls", "obj*cls"}:
            self.cfg.loss_target = lambda obj, cls: obj * cls
        else:
            raise NotImplementedError(
                f"Loss target {loss_target} not been implemented")

        # Generate init patch
        supported_modes = {"L", "RGB"}
        if self.cfg.patch_img_mode not in supported_modes:
            raise NotImplementedError(f"Currently only {supported_modes} channels supported")
        if self.cfg.patch_src == 'gray':
            adv_patch_cpu = self.generate_patch("gray", self.cfg.patch_img_mode)
        elif self.cfg.patch_src == 'random':
            adv_patch_cpu = self.generate_patch("random", self.cfg.patch_img_mode)
        else:
            adv_patch_cpu = self.read_image(self.cfg.patch_src, self.cfg.patch_img_mode)
        adv_patch_cpu.requires_grad = True

        optimizer = optim.Adam(
            [adv_patch_cpu], lr=self.cfg.start_lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=50)

        start_time = time.time()
        for epoch in range(1, self.cfg.n_epochs + 1):
            out_patch_path = osp.join(patch_dir, f"e_{epoch}.png")
            ep_loss = 0
            min_tv_loss = torch.tensor(self.cfg.min_tv_loss, device=self.dev)
            zero_tensor = torch.tensor([0], device=self.dev)

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(self.train_loader),
                                                        desc=f'Running train epoch {epoch}',
                                                        total=self.epoch_length):
                # with autograd.set_detect_anomaly(mode=True if self.cfg.debug_mode else False):
                img_batch = img_batch.to(self.dev, non_blocking=True)
                lab_batch = lab_batch.to(self.dev, non_blocking=True)
                adv_patch = adv_patch_cpu.to(self.dev, non_blocking=True)
                adv_batch_t = self.patch_transformer(
                    adv_patch, lab_batch, self.cfg.model_in_sz,
                    use_mul_add_gau=self.cfg.use_mul_add_gau,
                    do_transforms=self.cfg.transform_patches,
                    do_rotate=self.cfg.rotate_patches,
                    rand_loc=self.cfg.random_patch_loc)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (self.cfg.model_in_sz[0], self.cfg.model_in_sz[1]))

                if self.cfg.debug_mode:
                    img = p_img_batch[0, :, :, ]
                    img = T.ToPILImage()(img.detach().cpu())
                    img.save(osp.join(self.cfg.log_dir, "train_patch_applied_imgs", f"b_{i_batch}.jpg"))

                with autocast() if self.cfg.use_amp else nullcontext():
                    output = self.model(p_img_batch)[0]
                    max_prob = self.prob_extractor(output)
                    sal = self.sal_loss(adv_patch) if self.cfg.sal_mult != 0 else zero_tensor
                    nps = self.nps_loss(adv_patch) if self.cfg.nps_mult != 0 else zero_tensor
                    tv = self.tv_loss(adv_patch) if self.cfg.tv_mult != 0 else zero_tensor

                det_loss = torch.mean(max_prob)
                sal_loss = sal * 0
                nps_loss = nps * self.cfg.nps_mult
                tv_loss = torch.max(tv * self.cfg.tv_mult, min_tv_loss)

                loss = det_loss + nps_loss + tv_loss  #  + sal_loss
                ep_loss += loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # keep patch in cfg image pixel range
                pl, ph = self.cfg.patch_pixel_range
                adv_patch_cpu.data.clamp_(pl / 255, ph / 255)

                if i_batch % self.cfg.tensorboard_batch_log_interval == 0 and self.writer is not None and False:
                    iteration = self.epoch_length * epoch + i_batch
                    self.writer.add_scalar(
                        "total_loss", loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar(
                        "loss/det_loss", det_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar(
                        "loss/sal_loss", sal_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar(
                        "loss/nps_loss", nps_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar(
                        "loss/tv_loss", tv_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar(
                        "misc/epoch", epoch, iteration)
                    self.writer.add_scalar(
                        "misc/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                    self.writer.add_image(
                        "patch", adv_patch_cpu, iteration)
                if i_batch + 1 < len(self.train_loader):
                    del adv_batch_t, output, max_prob, det_loss, p_img_batch, sal_loss, nps_loss, tv_loss, loss
                    # torch.cuda.empty_cache()  # note emptying cache adds too much overhead
            ep_loss = ep_loss / len(self.train_loader)
            scheduler.step(ep_loss)

            # save patch after every patch_save_epoch_freq epochs
            if epoch % self.cfg.patch_save_epoch_freq == 0:
                img = T.ToPILImage(self.cfg.patch_img_mode)(adv_patch_cpu)
                img.save(out_patch_path)
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, sal_loss, nps_loss, tv_loss, loss
                # torch.cuda.empty_cache()  # note emptying cache adds too much overhead

            # run validation to calc asr on val set if self.val_dir is not None
            if all([self.cfg.val_image_dir, self.cfg.val_epoch_freq]) and epoch % self.cfg.val_epoch_freq == 0:
                with torch.no_grad():
                    self.val(epoch, out_patch_path)
        print(f"Total training time {time.time() - start_time:.2f}s")

    def val(self, epoch: int, patchfile: str, conf_thresh: float = 0.4, nms_thresh: float = 0.4) -> None:
        """
        Calculates the attack success rate according for the patch with respect to different bounding box areas
        """
        # load patch from file
        patch_img = Image.open(patchfile).convert(self.cfg.patch_img_mode)
        patch_img = T.Resize(self.cfg.patch_size)(patch_img)
        adv_patch_cpu = T.ToTensor()(patch_img)
        adv_patch = adv_patch_cpu.to(self.dev)

        img_paths = glob.glob(osp.join(self.cfg.val_image_dir, "*"))
        img_paths = sorted([p for p in img_paths if osp.splitext(p)[-1] in IMG_EXTNS])

        train_t_size_frac = self.patch_transformer.t_size_frac
        self.patch_transformer.t_size_frac = [0.3, 0.3]  # use a frac of 0.3 for validation
        # to calc confusion matrixes and attack success rates later
        all_labels = []
        all_patch_preds = []

        m_h, m_w = self.cfg.model_in_sz
        cls_id = self.cfg.objective_class_id
        zeros_tensor = torch.zeros([1, 5]).to(self.dev)
        #### iterate through all images ####
        for imgfile in tqdm(img_paths, desc=f'Running val epoch {epoch}'):
            img_name = osp.splitext(imgfile)[0].split('/')[-1]
            img = Image.open(imgfile).convert('RGB')
            padded_img = pad_to_square(img)
            padded_img = T.Resize(self.cfg.model_in_sz)(padded_img)

            #######################################
            # generate labels to use later for patched image
            padded_img_tensor = T.ToTensor()(padded_img).unsqueeze(0).to(self.dev)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            all_labels.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            labels = []
            for box in boxes:
                cls_id_box = box[-1].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                labels.append([cls_id_box, x_center / m_w, y_center / m_h, width / m_w, height / m_h])

            # save img if debug mode
            if self.cfg.debug_mode:
                padded_img_drawn = PatchTester.draw_bbox_on_pil_image(
                    all_labels[-1], padded_img, self.cfg.class_list)
                padded_img_drawn.save(osp.join(self.cfg.log_dir, "val_clean_imgs", img_name + ".jpg"))

            # use a filler zeros array for no dets
            label = np.asarray(labels) if labels else np.zeros([1, 5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            #######################################
            # Apply proper patches
            img_fake_batch = padded_img_tensor
            lab_fake_batch = label.unsqueeze(0).to(self.dev)

            if len(lab_fake_batch[0]) == 1 and torch.equal(lab_fake_batch[0], zeros_tensor):
                # no det, use images without patches
                p_tensor_batch = padded_img_tensor
            else:
                # transform patch and add it to image
                adv_batch_t = self.patch_transformer(
                    adv_patch, lab_fake_batch, self.cfg.model_in_sz,
                    use_mul_add_gau=self.cfg.use_mul_add_gau,
                    do_transforms=self.cfg.transform_patches,
                    do_rotate=self.cfg.rotate_patches,
                    rand_loc=self.cfg.random_patch_loc)
                p_tensor_batch = self.patch_applier(img_fake_batch, adv_batch_t)

            pred = self.model(p_tensor_batch)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            all_patch_preds.append(boxes.clone())

            # save properly patched img if debug mode
            if self.cfg.debug_mode:
                p_img_pil = T.ToPILImage('RGB')(p_tensor_batch.squeeze(0).cpu())
                p_img_pil_drawn = PatchTester.draw_bbox_on_pil_image(
                    all_patch_preds[-1], p_img_pil, self.cfg.class_list)
                p_img_pil_drawn.save(osp.join(self.cfg.log_dir, "val_patch_applied_imgs", img_name + ".jpg"))

        # reorder labels to (Array[M, 5]), class, x1, y1, x2, y2
        all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
        # patch and noise labels are of shapes (Array[N, 6]), x1, y1, x2, y2, conf, class
        all_patch_preds = torch.cat(all_patch_preds)
        asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(
            all_labels, all_patch_preds,
            class_list=self.cfg.class_list,
            cls_id=cls_id)

        print(f"Validation metrics for images with patches:")
        print(f"\tASR@thres={conf_thresh}: asr_s={asr_s:.3f},  asr_m={asr_m:.3f},  asr_l={asr_l:.3f},  asr_a={asr_a:.3f}")

        # self.writer.add_scalar("val_asr_per_epoch/area_small", asr_s, epoch)
        # self.writer.add_scalar("val_asr_per_epoch/area_medium", asr_m, epoch)
        # self.writer.add_scalar("val_asr_per_epoch/area_large", asr_l, epoch)
        # self.writer.add_scalar("val_asr_per_epoch/area_all", asr_a, epoch)
        # del adv_batch_t, padded_img_tensor, p_tensor_batch
        torch.cuda.empty_cache()
        self.patch_transformer.t_size_frac = train_t_size_frac


def main():
    parser = get_argparser()
    args = parser.parse_args()
    cfg = load_config_object(args.config)

    cfg["seed"] = args.seed
    cfg["weights_file"] = cfg["weights_file"].replace("placeholder", f"{args.name}")
    cfg["log_dir"] = cfg["log_dir"].replace("placeholder", f"{args.name}_{args.seed}")


    # optionally set seed for repeatability
    SEED = cfg["seed"]
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

    trainer = PatchTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
