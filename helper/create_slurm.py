import os

exp_list = [
    ("_detect", 1, 0, 1, "patch_det"),
    ("_detect", 0, 1, 1, "adv_det"),
    ("", 0, 0, 0, "naive"),
    ("", 0, 1, 0, "adv"),
    ("", 1, 0, 0, "patch"),
]

c = 0
for seed in range(5):
    for dataset_suffix, apply_patch, apply_adv, detect, name in exp_list:
        # os.system(f"rm run_{c}.sh")
        with open(f"run_{c}.sh", "w") as f:
            f.write(
                f'''#!/bin/sh
#SBATCH --job-name={c}_test_job
#SBATCH --out="log_run/my_script.out_{c}.txt"
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:tesla_a100:1
python train.py --data FLIR{dataset_suffix}.yaml --weights yolov5s.pt --img 640 --batch-size 8 --apply-adv {apply_adv} --apply-patch {apply_patch} --detect-patch {detect} --seed {seed + 1} --name {name}_{seed + 1}
python train_patch.py --cfg adv_patch_gen/configs/{name}{seed + 1}.json
python test_patch_2.py --save-img --target-class 0 --sd test_{name}_{seed + 1} --id /home/fmg/v-strack/datasets/FLIR/images/val/ -p /home/fmg/v-strack/yolov5_adversarial/runs/train_adversarial/{name}_{seed + 1}/patches0/e_50.png -w /home/fmg/v-strack/yolov5_adversarial/runs/train/{name}_{seed + 1}/weights/best.pt --cfg /home/fmg/v-strack/yolov5_adversarial/adv_patch_gen/configs/{name}{seed + 1}.json
cd HOTCOLDBlock
python main.py --batch_size 16 --weights /home/fmg/v-strack/yolov5_adversarial/runs/train/{name}_{seed + 1}/weights/best.pt --data /home/fmg/v-strack/yolov5_adversarial/HOTCOLDBlock/victim_detector/data/custom.yaml
''')
        os.system(f"sbatch run_{c}.sh")
        c += 1
