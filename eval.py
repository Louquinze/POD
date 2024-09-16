import os

id = 1
name = "naive"
version = "clean"

for id in [1,2,3,4,5]:
    for name in ["naive", "adv", "patch", "adv_det", "patch_det"]:  #
        try:
            os.remove("e_2.sh")
        except:
            pass
        with open("e_2.sh", "a") as f:
            f.write(f"python create_patch_dataset.py --save-img --target-class 0 --sd {name}_{id} --id ../datasets/FLIR/images/val/ -p runs/train_adversarial/{name}_{id}/patches0/e_50.png -w runs/train/{name}_{id}/weights/best.pt --cfg adv_patch_gen/configs/{name}{id}.json\n")
        os.system("bash e_2.sh")
        for version in ["clean", "proper_patched", "random_patched"]:
            print(os.getcwd())
            try:
                os.remove("e.sh")
            except:
                pass

            if "det" in name:
                add = "detect_"
            else:
                add = ""
            with open("e.sh", "a") as f:
                f.write(
                    f"""rm /home/lukas/PycharmProjects/AttackExp/datasets/FLIR_patch/images/val/*
            cp /home/lukas/PycharmProjects/AttackExp/yolov5_adversarial/{name}_{id}/base/{version}/images/* /home/lukas/PycharmProjects/AttackExp/datasets/FLIR_patch/images/val/
            python val.py --data data/FLIR_{add}patch.yaml --weights runs/train/{name}_{id}/weights/best.pt --name {name}_{id}_{version} --target-class 0 --device cuda:0 --batch-size 1
                    """
                )

            os.system("bash e.sh")
        os.system("bash e_a.sh")