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

from utils import set_gpu, seed_everything, load_yaml
from preprocessing import train_transform, test_transform, test_transform
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
    model = smp.UnetPlusPlus(encoder_name=encoder_type, encoder_weights='noisy-student', classes=1) # [imagenet, noisy-student]
    ckpt = torch.load(path)
    ckpt = OrderedDict({k.replace('module.encoder.',''):v for k,v in ckpt.items()})
    model.load_state_dict(ckpt) # cpu시에 torch.load(path, map_location='cpu')
    return model

def get_model_paths(dir_name):
    """
    checkpoint 3개 ensemble (가장 높은 score, 전후 epoch ckpt 사용)
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

def test(model, data_loader, device, step_num, save_to, TASK:str, restore=True):
    print(f'Start Inference of {TASK} Test Set')
    model.eval()
    with torch.no_grad():
        for step, (images, fnames, sizes) in enumerate(data_loader, 1):
            printProgress(step, step_num, 'Progress:', 'Complete', 1, 50)
            images = torch.stack(images)  # (N, C, W, H)
            images = images.to(device)

            model = model.to(device)
            outputs = model(images) # (N, C, W, H)

            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu() # 학습시 실행
            outputs = outputs.permute([0,2,3,1]) # (N, W, H, C)
            outputs = nn.functional.sigmoid(outputs.squeeze()).unsqueeze(-1)
            outputs = np.where(outputs[:,:,:,0]>=.5, 1, 0) # (N, W, H)

            if restore:
                outputs = np.expand_dims(outputs, -1)
                outputs = [np.squeeze(A.Resize(h, w, always_apply=True)(image=output.astype(np.float32))['image'])
                        for output, (w, h) in zip(outputs, sizes)]

            save_preds(outputs, save_to, fnames)

def test_ensemble(model, data_loader, device, step_num, save_to, TASK:str, restore=True):
    print(f'Start Inference of {TASK} Test Set')
    with torch.no_grad():
        for step, (images, fnames, sizes) in enumerate(data_loader, 1):
            printProgress(step, step_num, 'Progress:', 'Complete', 1, 50)
            images = torch.stack(images)  # (N, C, W, H)
            images = images.to(device)

            if isinstance(model, list):
                models=model
                output_list = []   
                for m in models:
                    m.eval()
                    m = m.to(device)
                    outputs = m(images) # (N, C, W, H)
                    output_list.append(outputs)        
                outputs = sum(output_list)/len(output_list)
            else:
                model.eval()
                model = model.to(device)
                outputs = model(images) # (N, C, W, H)

            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu() # 학습시 실행
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

def generate_testset_predictions(test_a2c_path, batch_size, encoder_type):
    test_a2c_path = './heartdata/test/A2C'

    test_a2c_df = make_data_df(test_a2c_path)

    test_a2c_dataset = CustomDataset(df=test_a2c_df, mode='test', transform=test_transform(768), has_masks=False)
    test_a2c_loader = torch.utils.data.DataLoader(
                            dataset=test_a2c_dataset, 
                            batch_size=batch_size,
                            num_workers=4,
                            collate_fn=collate_fn,
                            )

    a2c_paths = [
                './checkpoints/timm-efficientnet-b6-UnetPlusPlus(pre)_ep17_vloss0.0198_vdsc0.9449_vji0.8977_all.pth',
                './checkpoints/timm-efficientnet-b6-UnetPlusPlus(pre)_ep18_vloss0.019_vdsc0.9483_vji0.9039_all.pth'
                ]
    
    print('a2c_ckpts:',a2c_paths)

    a2c_models = list(map(lambda x: load_model(x, encoder_type), a2c_paths))
    
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
            a2c_models = list(map(nn.DataParallel, a2c_models))

    a2csteps = (len(test_a2c_dataset)//batch_size)+1
    a2c_save_to = './test_pred/A2C/'

    test_ensemble(a2c_models, test_a2c_loader, device, a2csteps, a2c_save_to, 'A2C', restore=True)
    
    print('Generated Predictions of Test Set !!!')

def test_score_pred_and_mask(test_a2c_mask_path, test_a2c_pred_path, batch_size):

    # A2C 정답 및 예측 Mask 파일들의 경로 DataFrame 생성    
    test_a2c_mask_df = make_mask_df(test_a2c_mask_path)
    test_a2c_pred_df = make_mask_df(test_a2c_pred_path)

    test_a2c_score_dataset = ScoreDataset(mask_df=test_a2c_mask_df,
                                        pred_df=test_a2c_pred_df)
    test_a2c_score_loader = torch.utils.data.DataLoader(
                            dataset=test_a2c_score_dataset, 
                            batch_size=batch_size,
                            num_workers=4,
                            collate_fn=collate_fn,
                            )

    a2c_cnt = len(test_a2c_score_dataset)
    a2c_dsc_sum = 0.
    a2c_ji_sum = 0.

    for a2c_masks, a2c_preds in test_a2c_score_loader:

        # Batch의 A2C DSC, JI score 가산
        dsc_li = m_dsc_bin(a2c_masks, a2c_preds, return_single=True)
        ji_li = m_ji_bin(a2c_masks, a2c_preds, return_single=True)

        a2c_dsc_sum+=sum(dsc_li)
        a2c_ji_sum+=sum(ji_li)

    # 전체 A2C DSC, JI score 계산
    a2c_dsc = a2c_dsc_sum/a2c_cnt
    a2c_ji = a2c_ji_sum/a2c_cnt

    print(f'<A2C Test set> DSC : {round(a2c_dsc, 4)}, JI: {round(a2c_ji, 4)}') 
    
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
    ENCODER_TYPE = 'timm-efficientnet-b6'
    BATCH_SIZE = 12     

    # Test Set 예측 Mask 생성 및 저장
    test_a2c_path = './heartdata/test/A2C'

    generate_testset_predictions(test_a2c_path, BATCH_SIZE, ENCODER_TYPE)

    # Test Set 예측 Mask와 정답 Mask 간 Score 계산
    test_a2c_mask_path = './heartdata/test/A2C'
    test_a2c_pred_path = './test_pred/A2C'

    # test_score_pred_and_mask(test_a2c_mask_path, test_a2c_pred_path, BATCH_SIZE)

