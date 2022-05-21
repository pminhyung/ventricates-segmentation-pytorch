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
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

from datasets import CustomDataset, ScoreDataset, make_data_df, make_mask_df, collate_fn
from preprocessing import train_transform, val_transform, test_transform
from utils import set_gpu, seed_everything, make_savedir, printProgress
from metrics import m_dsc_bin, m_ji_bin

# Fix Seed
SEED=21
seed_everything(SEED)    

BATCH_SIZE = 12 

# Choose GPUs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NUM = torch.cuda.device_count()
if GPU_NUM:
    gpu_li = [_ for _ in range(GPU_NUM)]
    set_gpu(gpu_li)


def set_global_val_vars(args):

    # Set Global Variables (Hyperparameters)
    globals().update(
                    dict(
                        ENCODER_TYPE = ('timm-efficientnet-b6' if args.task=='a2c'
                                        else 'timm-efficientnet-b5')
                        )
                    )


def get_model_path(dir_name):
    ckpt_paths = glob(dir_name+'/*_all.*')
    sort_by_max_ji = lambda x : float(x.split('vji')[1].replace('_all.pth',''))
    model_path = sorted(ckpt_paths, key = sort_by_max_ji, reverse=True)[0]
    return model_path

def get_model_paths(dir_name, num_model):
    """
    checkpoint ensemble (가장 높은 score, 전후 epoch ckpt 사용)
    """
    ckpt_paths = glob(dir_name+'/*_all.*')
    sort_by_epoch = lambda x : re.findall('ep[0-9]+', x)[0].replace('ep','')
    sort_by_max_ji = lambda x : float(x.split('vji')[1].replace('_all.pth',''))
    model_path = sorted(ckpt_paths, key = sort_by_max_ji, reverse=True)[0]
    ckpt_paths = sorted(ckpt_paths, key = sort_by_epoch) # 오름차순
    idx = ckpt_paths.index(model_path)
    model_paths = ckpt_paths[idx-(num_model//2):idx+(num_model - num_model//2)]
    return model_paths

def load_model(path, encoder_type):
    encoder_weights = 'imagenet' if encoder_type== 'timm-efficientnet-b8' else 'noisy-student'
    model = smp.UnetPlusPlus(encoder_name=encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
    ckpt = torch.load(path)
    ckpt = OrderedDict({k.replace('module.encoder.',''):v for k,v in ckpt.items()})
    model.load_state_dict(ckpt) # cpu시에 torch.load(path, map_location='cpu')
    return model

def save_preds(outputs, save_to, fnames):
    make_savedir(save_to)
    for pred, fname in zip(outputs, fnames):
        basename = os.path.basename(fname)
        np.save(save_to+basename.replace('png', 'npy'), pred)

def validation(args, model, data_loader, device, step_num, save_to, restore=True):
    print(f'Start Inference of {args.task.upper()} Validation Set')

    with torch.no_grad():
        for step, (images, fnames, sizes) in enumerate(data_loader, 1):
            printProgress(step, step_num, 'Progress:', 'Complete', 1, 50)
            images = torch.stack(images)  # (N, C, W, H)
            images = images.to(device)

            if isinstance(model, list):
                models = model
                output_list = []

                # ensemble : soft voting   
                for m in models:
                    m.eval()
                    m = m.to(device)
                    outputs = m(images) # (N, C, W, H)
                    output_list.append(outputs)        
                outputs = sum(output_list)/len(output_list)
            else:
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

def generate_valset_predictions(args, val_path, batch_size, encoder_type):

    val_df = make_data_df(val_path)

    val_dataset = CustomDataset(df=val_df, mode='test', transform=val_transform(768), has_masks=False)
    val_loader = torch.utils.data.DataLoader(
                            dataset=val_dataset, 
                            batch_size=batch_size,
                            num_workers=4,
                            collate_fn=collate_fn,
                            )
    
    if args.task == 'a2c':
        # paths = get_model_paths(f'./saved/{args.task}', num_model = 3)
        paths = [
                './checkpoints/timm-efficientnet-b6-UnetPlusPlus(pre)_ep17_vloss0.0198_vdsc0.9449_vji0.8977_all.pth',
                './checkpoints/timm-efficientnet-b6-UnetPlusPlus(pre)_ep18_vloss0.019_vdsc0.9483_vji0.9039_all.pth'
                ]
        models = list(map(lambda x: load_model(x, encoder_type), paths))

        print(f'{args.task}_ckpts:', paths)

        if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
            model = list(map(nn.DataParallel, models))

    else:
        path = './checkpoints/timm-efficientnet-b5-UnetPlusPlus(pre)_ep16_vloss0.0159_vdsc0.964_vji0.9312_all.pth'

        print(f'{args.task}_ckpt:', paths)

        model = load_model(path, encoder_type)

        if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:        
            model = nn.DataParallel(model)

    steps = (len(val_dataset)//batch_size)+1
    save_to = f'./val_pred/{args.task.upper()}/'

    validation(args, model, val_loader, DEVICE, steps, save_to, restore=True)
    
    print('Generated Predictions of Validation Set !!!')

def val_score_pred_and_mask(val_mask_path, val_pred_path, batch_size):

    # 정답 및 예측 Mask 파일들의 경로 DataFrame 생성    
    val_mask_df = make_mask_df(val_mask_path)
    val_pred_df = make_mask_df(val_pred_path)

    val_score_dataset = ScoreDataset(mask_df=val_mask_df,
                                        pred_df=val_pred_df)

    val_score_loader = torch.utils.data.DataLoader(
                            dataset=val_score_dataset, 
                            batch_size=batch_size,
                            num_workers=4,
                            collate_fn=collate_fn,
                            )

    cnt = len(val_score_dataset)
    dsc_sum = 0.
    ji_sum = 0.

    for masks, preds in val_score_loader:

        # Batch의 DSC, JI score 가산
        dsc_li = m_dsc_bin(masks, preds, return_single=True)
        ji_li = m_ji_bin(masks, preds, return_single=True)

        dsc_sum+=sum(dsc_li)
        ji_sum+=sum(ji_li)

    # 전체 DSC, JI score 계산
    dsc = dsc_sum/cnt
    ji = ji_sum/cnt

    print(f'<{args.task.upper()} Validation set> DSC : {round(dsc, 4)}, JI: {round(ji, 4)}') 

def main(args):

    set_global_val_vars(args)

    # Validation Set 예측 Mask 생성 및 저장
    val_path = f'./heartdata/validation/{args.task}'

    generate_valset_predictions(val_path, BATCH_SIZE, ENCODER_TYPE)

    # Validation Set 예측 Mask와 정답 Mask 간 Score 계산
    val_mask_path = f'./heartdata/validation/{args.task}'
    val_pred_path = f'./val_pred/{args.task}'

    val_score_pred_and_mask(val_mask_path, val_pred_path, BATCH_SIZE)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--task', type=str, choices = ['a2c', 'a4c'])
    args = parser.parse_args()

    main(args)

    

