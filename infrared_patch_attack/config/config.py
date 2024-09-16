import torch
import os
import time

class Config(object):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    display = False
    iterations = 5
    width, height = 30, 60
    # width, height = 50, 100
    emp_iterations = 100
    # max_pertubation_mask = 50
    cover_rate = 0.1
    max_pertubation_mask = int(width * height * cover_rate * 2 / 3)
    content = 0  # range(0, 1, 0.1)
    grad_avg = False # DI
    res_folder = "res"
    attack_dataset_folder = "datasets/pedestrian2"
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    save_folder = os.path.join(res_folder, time.asctime( time.localtime(time.time()) ))
    save_folder = save_folder.replace(":","")

