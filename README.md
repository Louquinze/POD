
# POD: Patch-based Occlusion-aware Detection

Infrared detection is a promising technique for safety-critical tasks due to its excellent anti-interference capabilities. However, recent research highlights its vulnerability to physically realizable adversarial patches, posing significant risks in real-world applications.

To address this issue, we present the first comprehensive study on defense strategies against adversarial patch attacks in infrared detection, specifically focusing on human detection. Our novel defense strategy, Patch-based Occlusion-aware Detection (POD), enhances training samples with random patches and subsequently detects them. POD not only robustly detects people but also identifies the locations of adversarial patches. Despite its computational efficiency, POD generalizes well to state-of-the-art adversarial patch attacks unseen during training. Additionally, POD improves detection precision even in clean (i.e., non-attack) scenarios due to the data augmentation effect. Our evaluation demonstrates that POD is robust to adversarial patches of various shapes and sizes, providing a viable defense mechanism for real-world infrared human detection systems and paving the way for future research.

[Visit our paper: Defending Against Physical Adversarial Patch Attacks on Infrared Human Detection](https://arxiv.org/abs/2309.15519v3)

## Installation & Setup

Clone this repository. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Download the datasets and place the files in a folder `datasets` one level above this repo.

```
├───datasets
│   ├───CVC-Sub
│   │   └───valid
│   │       ├───images
│   │       └───labels
│   ├───FLIR
│   │   ├───images
│   │   │   ├───test
│   │   │   ├───train
│   │   │   └───val
│   │   └───labels
│   │       ├───test
│   │       ├───train
│   │       └───val
│   └───Pedestrian-CVC-14
│       └───train
│           ├───images
│           └───labels
└───yolov5_adversarial
    ├───adv_patch_gen
    ├───data
    ├───helper
    ├───HOTCOLDBlock
    │   ├───adv_patch_gen
    │   │   ├───configs
   [..][..]
    ├───infrared_patch_attack
    │   ├───attack
    [..][..]
```
For the datasets please download the data [here](...)

## Running Experiments

### YAML and JSON Configuration Files

Experiments are configured through YAML and JSON files:

- **YAML files** (e.g., `FLIR_detect.yaml`, `FLIR.yaml`, `FLIR_val_patch.yaml`, `FLIR_val_detect_patch.yaml`):
  - Define the dataset and experiment settings.
  - Ensure the configuration matches the suffix `_detect.yaml` when detection is enabled.

- **JSON files** (e.g., `adv_patch_gen/configs/default.json`, `adv_patch_gen/configs/default_no_patch.json`):
  - Used for patch generation and adversarial training configurations.

### Training the Model

To train the model with various configurations, use the following command:

```bash
python train.py --data <path_to_yaml> --weights <path_to_weights> --img <int:image_size> --batch-size <int> --apply-adv <0_or_1> --apply-patch <0_or_1> --detect-patch <0_or_1> --seed <int> --name <str:meaningful_name> --epochs <int> --patch_selection <selection_encoding>
```

**Arguments:**

- `--apply-adv`: Apply adversarial training (0: No, 1: Yes).
- `--apply-patch`: Apply patches during training (0: No, 1: Yes).
- `--detect-patch`: Enable a second class for patch detection (0: No, 1: Yes).
- `--name`: A meaningful name for the experiment (used in file paths).
- `patch_selection`: Defines which augmentation to use. Choose between `cutout`, `noise`, and `inv` (and `geometry`, not present in the paper). Specify them separated by `-`, like `--patch_selection inv-cutout` or `--patch_selection inv`.

#### Example

```bash
python train.py --data FLIR_detect.yaml --weights yolov5s.pt --img 640 --batch-size 2 --apply-adv 0 --apply-patch 1 --detect-patch 1 --seed 1 --name pod_detect --epochs 50 --patch_selection cutout-noise-inv
```

### Training the Patch

To train the adversarial patch, use the following command:

```bash
python train_patch.py --cfg <path_to_json_config> --batch_size <int> --name <str:meaningful_name>_<int:patch_num> --seed <int>
```

#### Example

```bash
python train_patch.py --cfg adv_patch_gen/configs/default.json --batch_size 8 --name pod_detect_1 --seed 42
```

### Evaluating the Patch

To evaluate the effectiveness of the patch, run:

```bash
python test_patch.py --save-img --target-class 0 --sd test/<str:meaningful_name>_<int:patch_num>_<int:seed> --id <path_to_images> --cfg <path_to_json_config> -p <path_to_patch_image> -w <path_to_model_weights>
```

**Arguments:**

- `--save-img`: Save the output images.
- `--target-class`: The target class for patch evaluation (e.g., 0 for background, 1 for person).
- `--sd`: Save directory for the results.
- `--id`: Path to the validation images.
- `--cfg`: Path to the JSON configuration file.
- `--p`: Path to the patch image.
- `--w`: Path to the trained model weights.

#### Example

```bash
python test_patch.py --save-img --target-class 0 --sd test/pod_detect_1_1 --id ../datasets/FLIR/images/val/ -p ./runs/train_adversarial/pod_detect_1_1/patches0/e_50.png -w ./runs/train/pod_detect_1/weights/best.pt --cfg ./adv_patch_gen/configs/default.json
```

## HOTCOLDBlock

To experiment with HOTCOLDBlock, navigate to the directory:

```bash
cd HOTCOLDBlock
```

### Running Experiments

```bash
python main.py --batch_size <int> --weights <path_to_weights> --data <path_to_yaml>
```

#### Example

```bash
python main.py --batch_size 16 --weights ../runs/train/pod_detect_1/weights/best.pt --data ./victim_detector/data/custom.yaml
```

## Infrared Patch Attack

Navigate to the `infrared_patch_attack` directory:

```bash
cd infrared_patch_attack
```

### Running Experiments

```bash
python shaped_patch_attack.py --model <str:meaningful_name>_<int:seed>
```

#### Example

```bash
python shaped_patch_attack.py --model pod_detect_1
```

## 3. Authors
- Lukas Strack
- Futa Waseda (https://scholar.google.co.jp/citations?user=aBQ2en8AAAAJ&hl=en)
- Huy H. Nguyen (https://researchmap.jp/nhhuy/?lang=english)
- Zheng Yinqiang [(https://researchmap.jp/nhhuy/?lang=english](https://researchmap.jp/yinqiangzheng?lang=en)
- Isao Echizen (https://researchmap.jp/echizenisao/?lang=english)


## Reference
Strack, L., Waseda, F., Nguyen, H. H., Zheng, Y., & Echizen, I. (2023). Defending Against Physical Adversarial Patch Attacks on Infrared Human Detection (Version 3). arXiv. https://doi.org/10.48550/ARXIV.2309.15519