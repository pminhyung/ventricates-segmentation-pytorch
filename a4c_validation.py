import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
import sys
import os
import re
from glob import glob
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

from utils import set_gpu, seed_everything
from preprocessing import train_transform, val_transform, test_transform
from datasets import CustomDataset, ScoreDataset, make_data_df, make_mask_df, collate_fn
from metrics import m_dsc_bin, m_ji_bin

def make_savedir(saved_dir):
    directory = saved_dir
    stack = []
    while directory!='.':
        stack.append(directory)
        directory = os.path.dirname(directory)

    while stack:
        subdir = stack.pop()
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

def get_model_path(dir_name):
    ckpt_paths = glob(dir_name+'/*_all.*')
    sort_by_max_ji = lambda x : float(x.split('vji')[1].replace('_all.pth',''))
    model_path = sorted(ckpt_paths, key = sort_by_max_ji, reverse=True)[0]
    return model_path

def load_model(path, encoder_type):
    encoder_weights = 'imagenet' if encoder_type== 'timm-efficientnet-b8' else 'noisy-student'
    model = smp.UnetPlusPlus(encoder_name=encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
    ckpt = torch.load(path)
    ckpt = OrderedDict({k.replace('module.encoder.',''):v for k,v in ckpt.items()})
    model.load_state_dict(ckpt) # cpu시에 torch.load(path, map_location='cpu')
    return model

def get_model_paths(dir_name):
    """
    checkpoint k개 ensemble (가장 높은 score, 전후 epoch ckpt 사용)
    """
    
    re.search('[0-9]+', '123abc')
    ckpt_paths = glob(dir_name+'/*_all.*')
    sort_by_epoch = lambda x : re.findall('ep[0-9]+', x)[0].replace('ep','')
    sort_by_max_ji = lambda x : float(x.split('vji')[1].replace('_all.pth',''))
    model_path = sorted(ckpt_paths, key = sort_by_max_ji, reverse=True)[0]
    ckpt_paths = sorted(ckpt_paths, key = sort_by_epoch) # 오름차순
    idx = ckpt_paths.index(model_path)
    model_paths = ckpt_paths[idx-1:idx+2]
    return model_paths

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100): 
    formatStr = "{0:." + str(decimals) + "f}" 
    percent = formatStr.format(100 * (iteration / float(total))) 
    filledLength = int(round(barLength * iteration / float(total))) 
    bar = '#' * filledLength + '-' * (barLength - filledLength) 
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)), 
    if iteration == total: 
        sys.stdout.write('\n') 
    sys.stdout.flush()

def save_preds(outputs, save_to, fnames):

    make_savedir(save_to)

    for pred, fname in zip(outputs, fnames):
        basename = os.path.basename(fname)
        np.save(save_to+basename.replace('png', 'npy'), pred)

def validation(model, data_loader, device, step_num, save_to, TASK:str, restore=True):
    print(f'Start Validation of {TASK} Test Set')
    model.eval()
    with torch.no_grad():
        for step, (images, fnames, sizes) in enumerate(data_loader, 1):
            printProgress(step, step_num, 'Progress:', 'Complete', 1, 50)
            images = torch.stack(images)  # (N, C, W, H)
            images = images.to(device)

            # device 할당
            model = model.to(device)
            
            outputs = model(images) # (N, C, W, H)

            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu() 
            outputs = outputs.permute([0,2,3,1]) # (N, W, H, C)
            outputs = nn.functional.sigmoid(outputs.squeeze()).unsqueeze(-1)
            outputs = np.where(outputs[:,:,:,0]>=.5, 1, 0) # (N, W, H)

            if restore:
                outputs = np.expand_dims(outputs, -1)
                outputs = [np.squeeze(A.Resize(h, w, always_apply=True)(image=output.astype(np.float32))['image'])
                        for output, (w, h) in zip(outputs, sizes)]

            save_preds(outputs, save_to, fnames)

def get_model_path(dir_name):
    ckpt_paths = glob(dir_name+'/*_all.*')
    sort_by_max_ji = lambda x : float(x.split('vji')[1].replace('_all.pth',''))
    model_path = sorted(ckpt_paths, key = sort_by_max_ji, reverse=True)[0]
    return model_path

def generate_valset_predictions(val_a4c_path, batch_size, encoder_type):
    
    val_a4c_df = make_data_df(val_a4c_path)

    val_a4c_dataset = CustomDataset(df=val_a4c_df, mode='test', transform=val_transform(512), has_masks=False)
    val_a4c_loader = torch.utils.data.DataLoader(
                            dataset=val_a4c_dataset, 
                            batch_size=batch_size,
                            num_workers=4,
                            collate_fn=collate_fn,
                            )

    a4c_path = './checkpoints/timm-efficientnet-b5-UnetPlusPlus(pre)_ep16_vloss0.0159_vdsc0.964_vji0.9312_all.pth'
    print('a4c_ckpt:',a4c_path)
    model_a4c = load_model(a4c_path, encoder_type)
    
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:        
        model_a4c = nn.DataParallel(model_a4c)

    a4csteps = (len(val_a4c_dataset)//batch_size)+1
    a4c_save_to = './val_pred/A4C/'

    validation(model_a4c, val_a4c_loader, device, a4csteps, a4c_save_to, 'A4C', restore=True)
    
    print('Generated Predictions of A4C Validation Set !!!')

def val_score_pred_and_mask(val_a4c_mask_path, val_a4c_pred_path, batch_size):

    # A4C 정답 및 예측 Mask 파일들의 경로 DataFrame 생성
    val_a4c_mask_df = make_mask_df(val_a4c_mask_path)
    val_a4c_pred_df = make_mask_df(val_a4c_pred_path)

    val_a4c_score_dataset = ScoreDataset(mask_df=val_a4c_mask_df,
                                        pred_df=val_a4c_pred_df)
    val_a4c_score_loader = torch.utils.data.DataLoader(
                            dataset=val_a4c_score_dataset, 
                            batch_size=batch_size,
                            num_workers=4,
                            collate_fn=collate_fn,
                            )

    a4c_cnt = len(val_a4c_score_dataset)

    a4c_dsc_sum = 0.
    a4c_ji_sum = 0.

    for a4c_masks, a4c_preds in val_a4c_score_loader:
        # Batch의 A4C DSC, JI score 가산
        dsc_li = m_dsc_bin(a4c_masks, a4c_preds, return_single=True)
        ji_li = m_ji_bin(a4c_masks, a4c_preds, return_single=True)

        a4c_dsc_sum+=sum(dsc_li)
        a4c_ji_sum+=sum(ji_li)

    # 전체 A4C DSC, JI score 계산
    a4c_dsc = a4c_dsc_sum/a4c_cnt
    a4c_ji = a4c_ji_sum/a4c_cnt

    print(f'<A4C Validation set> DSC : {round(a4c_dsc, 4)}, JI: {round(a4c_ji, 4)}') 

if __name__ == '__main__':

    # Fix Seed
    random_seed=21
    seed_everything(random_seed)    

    # Choose GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()

    if gpu_num:
        gpu_li = [_ for _ in range(gpu_num)]
        set_gpu(gpu_li)

    # Config Parsing
    # CONFIG_PATH = './config.yaml'
    # config = load_yaml(CONFIG_PATH)
    ENCODER_TYPE = 'timm-efficientnet-b5'
    BATCH_SIZE = 12

    # Validation Set 예측 Mask 생성 및 저장
    val_a4c_path = './heartdata/validation/A4C'

    generate_valset_predictions(val_a4c_path, BATCH_SIZE, ENCODER_TYPE)

    # Validation Set 예측 Mask와 정답 Mask 간 Score 계산
    val_a4c_mask_path = './heartdata/validation/A4C'
    val_a4c_pred_path = './val_pred/A4C'

    val_score_pred_and_mask(val_a4c_mask_path, val_a4c_pred_path, BATCH_SIZE)

