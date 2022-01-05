import torch
import numpy as np
import random
import os
import yaml

def set_gpu(*args):
    global device
    devices = ','.join(list(map(str, args)))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]= "0" # "0,1,2"
    os.environ["CUDA_VISIBLE_DEVICES"]= devices # "0,1,2"

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
    print('Device:', device)  # 출력결과: cuda 
    print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 1 (GPU #2 한개 사용하므로)
    print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (GPU #2 의미)
    print(torch.cuda.get_device_name(0))

# seed 고정
def seed_everything(seed: int = 21):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

