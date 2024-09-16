from itertools import combinations

exp_list = []
items = ["cutout", "inv", "noise", "geometry"]

# Generate all unique combinations for 1, 2, and 3 elements
unique_combinations = []
for i in range(1, len(items) + 1):
    unique_combinations.extend(combinations(items, i))

for selection in unique_combinations:
    exp_list.append(("", "_detect", f"patch_det_{'_'.join(selection)}", 0, 1, 1, f"{'-'.join(selection)}"),)

"""
exp_list = [
    ("_no_patch", "_detect", 1, 0, 1, "patch_det"),
    ("_no_patch", "_detect", 0, 1, 1, "adv_det"),
    ("", "", 0, 0, 0, "naive"),
    ("", "", 0, 1, 0, "adv"),
    ("", "", 1, 0, 0, "patch"),
]
"""

exp_list += [
    # name, {apply_adv} {apply_patch} {detect_patch} c
    ("_no_patch", "", "naive", 0, 0, 0, "cutout"),
    ("_no_patch", "", "patch", 0, 1, 0, "cutout"),
    ("_no_patch", "", "adv", 1, 0, 0, "cutout"),
    ("", "_detect", "adv_det", 1, 0, 1, "cutout"),
]

for seed in range(5):
    for exp in exp_list:
        json_suffix, dataset_suffix, exp_name, apply_adv, apply_patch, detect_patch, patch_selection = exp
        with open(f"run_exp_{exp_name}_{seed}.sh", "w") as f:
            f.write(
            f"""#!/bin/sh
#SBATCH --job-name={seed}_test_job
#SBATCH --out="log_run/my_script.out_{seed}.txt"
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:tesla_a100:1
python train.py --data FLIR{dataset_suffix}.yaml --weights yolov5s.pt --img 640 --batch-size 8 --apply-adv {apply_adv} --apply-patch {apply_patch} --detect-patch {detect_patch}  --seed {seed} --name {exp_name} --epochs 50  --patch_selection {patch_selection}
python train_patch.py --cfg adv_patch_gen/configs/default{json_suffix}.json --batch_size 8 --name {exp_name}_{seed} --seed {seed}
python test_patch.py --save-img --target-class 0 --sd test/{exp_name}_{seed}_{seed} --id ../datasets/FLIR/images/val/ -p ./runs/train_adversarial/{exp_name}_{seed}_{seed}/patches0/e_50.png -w ./runs/train/{exp_name}_{seed}/weights/last.pt --cfg ./adv_patch_gen/configs/default{json_suffix}.json
cd HOTCOLDBlock
python main.py --batch_size 16 --weights ../runs/train/{exp_name}_{seed}/weights/last.pt --data ./victim_detector/data/custom.yaml
cd ..
python val_patch.py --data data/FLIR_val{dataset_suffix}_patch.yaml --weights runs/train/{exp_name}_{seed}/weights/last.pt --name {exp_name}_{seed} --target-class 0
"""
        )