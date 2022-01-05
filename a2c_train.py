import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
import warnings 
warnings.filterwarnings('ignore')

from utils import set_gpu, seed_everything, load_yaml
from preprocessing import train_transform, val_transform, test_transform
from datasets import CustomDataset, make_data_df, collate_fn
from model import Encoder
from callbacks import GradualWarmupSchedulerV2, EarlyStoppingScore
from metrics import m_dsc_bin, m_ji_bin

def get_scheduler(optimizer):
    if PRE_SCHEDULER=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=PRE_FACTOR, patience=PRE_PATIENCE, verbose=True, eps=PRE_EPS)
    elif PRE_SCHEDULER=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=PRE_T_MAX, eta_min=PRE_MIN_LR, last_epoch=-1)
    elif PRE_SCHEDULER=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=PRE_T_0, T_mult=1, eta_min=PRE_MIN_LR, last_epoch=-1)
    elif PRE_SCHEDULER=='GradualWarmupSchedulerV2':
        scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, PRE_COSINE_EPO)
        scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=PRE_WARMUP_FACTOR, total_epoch=PRE_WARMUP_EPO, after_scheduler=scheduler_cosine)
        scheduler=scheduler_warmup        
    return scheduler

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

def save_model(model, saved_dir, file_name):
    make_savedir(saved_dir)
    checkpoint = model.state_dict()
    ckpt_path = os.path.join(saved_dir, file_name)
    torch.save(checkpoint, ckpt_path)

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, file_postfix, val_every, device):
    print(f'Start training..')
    best_val_loss = 9999999
    early_stopping = EarlyStoppingScore(patience=es_patience, min_delta=es_min_delta)
    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks, fnames) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).float()  # (N, W, H), BCEWithLogitsLoss(target.float()필요)
            
            # # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device) # 학습시 실행
            
            # device 할당
            model = model.to(device) # 학습시 실행
            
            # inference
            outputs = model(images)

            # BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), masks) # (N, 1, W, H) -> (N, W, H)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu().numpy() # 학습시 실행
            outputs = np.where(outputs[:,0,:,:]>=.5, 1, 0) # (N, W, H)

            masks = masks.detach().cpu().numpy() # 학습시 실행
            
            mdsc = m_dsc_bin(masks, outputs)
            mji = m_ji_bin(masks, outputs)

            # step 주기에 따른 loss 출력
            if (step + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mDSC: {round(mdsc,4)}, mJI: {round(mji, 4)}')
            
        scheduler.step(epoch)
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_loss, vdsc, vji = validation(epoch + 1, model, val_loader, criterion, device)
            
            if val_loss < best_val_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_val_loss = val_loss
                fname = f'{ENCODER_TYPE}-{DECODER_TYPE}(pre)_ep{epoch+1}_vloss{round(val_loss.item(),4)}_vdsc{round(vdsc, 4)}_vji{round(vji,4)}'+file_postfix+'.pth'
                save_model(model, saved_dir, fname)
            elif vji >= 0.7: # ckpt 앙상블용
                save_model(model, saved_dir, fname)
                     
            early_stopping(vji)
            if early_stopping.early_stop:
                break
    
    best_val_loss, best_val_dsc, best_val_ji = best_val_loss.detach().cpu().item(), vdsc, vji
    return best_val_loss, best_val_dsc, best_val_ji

segmentation_classes = ['background', 'heart']

def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        total_loss = 0
        cnt = 0 # step cnt
        
        dsc = 0.
        ji = 0.
        data_cnt = 0 # single data cnt
        for step, (images, masks, fnames, _) in enumerate(data_loader):
            
            images = torch.stack(images)  # (N, C, W, H)
            masks = torch.stack(masks).float()  # (N, W, H), BCEWithLogitsLoss(target.float()필요)

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images) # (N, C, W, H)
            
            # BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), masks) # (N, 1, W, H) -> (N, W, H)
            
            total_loss += loss
            cnt += 1
            
            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu().numpy() # 학습시 실행
            outputs = np.where(outputs[:,0,:,:]>=.5, 1, 0) # (N, W, H)

            masks = masks.detach().cpu().numpy() # 학습시 실행
            
            _, sum_dsc = m_dsc_bin(masks, outputs, return_sum=True)
            _, sum_ji = m_ji_bin(masks, outputs, return_sum=True)

            dsc+=sum_dsc
            ji+=sum_ji
            data_cnt+=outputs.shape[0]
            if step==0:
                images = images.detach().cpu().numpy()

        mdsc = dsc/data_cnt
        mji = ji/data_cnt
        avg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avg_loss.item(), 4)}, mDSC : {round(mdsc, 4)}, \
                mJI: {round(mji, 4)}')
        
    return avg_loss, mdsc, mji

def make_dir(save_dir):
    if not os.path.isdir(save_dir):                                                           
        os.mkdir(save_dir)

def save_train_all_res_df(score):
    df = pd.DataFrame(score).T.reset_index()
    save_dir = './trainall'
    make_dir(save_dir)
    df.to_csv(f'./trainall/a2c-all-res.csv', index=False)
    return df

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
    
    # Hyperparameters
    batch_size = 12
    num_epochs = 1000
    segmentation_classes = ['background', 'heart']
    es_patience=5
    es_min_delta=1e-5
    val_every = 1
    num_epochs = 1000
    num_to_log = 10
    score = {}

    # Config Parsing
    CONFIG_PATH = './config.yaml'
    config = load_yaml(CONFIG_PATH)

    IMGSIZE = 768

    PRE_COSINE_EPO = config['PREPROCESSING']['cosine_epo']
    PRE_WARMUP_EPO = config['PREPROCESSING']['warmup_epo']
    PRE_FREEZE_EPO = config['PREPROCESSING']['freeze_epo']
    PRE_WARMUP_FACTOR = config['PREPROCESSING']['warmup_factor']
    PRE_ENCODER_TYPE = config['PREPROCESSING']['encoder_type']
    PRE_DECODER_TYPE = config['PREPROCESSING']['decoder_type']
    PRE_WEIGHT_DECAY = config['PREPROCESSING']['weight_decay']
    PRE_ENCODER_LR = config['PREPROCESSING']['encoder_lr']
    PRE_EPOCHS = PRE_COSINE_EPO + PRE_WARMUP_EPO + PRE_FREEZE_EPO 
    PRE_SCHEDULER = config['PREPROCESSING']['scheduler']

    ENCODER_TYPE = 'timm-efficientnet-b6'
    DECODER_TYPE = 'UnetPlusPlus'
    PRETRAINED = True

    # Dataset, Dataloader 
    train_a2c_path = './heartdata/train/A2C'
    val_a2c_path = './heartdata/validation/A2C'

    train_a2c_df = make_data_df(train_a2c_path)
    test_a2c_df = make_data_df(val_a2c_path)

    train_a2c_dataset = CustomDataset(df=train_a2c_df, mode='train', transform=train_transform(IMGSIZE))
    train_a2c_loader = torch.utils.data.DataLoader(
                            dataset=train_a2c_dataset, 
                            batch_size=batch_size,
                            num_workers=4,
                            collate_fn=collate_fn,
                            drop_last=True
                            )

    test_a2c_dataset = CustomDataset(df=test_a2c_df, mode='val', transform=test_transform(IMGSIZE))
    test_a2c_loader = torch.utils.data.DataLoader(
                            dataset=test_a2c_dataset, 
                            batch_size=batch_size,
                            num_workers=4,
                            collate_fn=collate_fn,
                            )

    # Model
    model_a2c = Encoder(ENCODER_TYPE, DECODER_TYPE, pretrained=PRETRAINED)

    # Optimizer
    optimizer_a2c = torch.optim.Adam(model_a2c.parameters(), lr=PRE_ENCODER_LR, weight_decay=PRE_WEIGHT_DECAY, amsgrad=False)

    # schedulers
    scheduler_a2c = get_scheduler(optimizer_a2c)

    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        model_a2c = nn.DataParallel(model_a2c)

    criterion = nn.BCEWithLogitsLoss()

    val_every = 1
    num_epochs = 1000
    num_to_log = 10
    score = {}

    saved_dir = './saved/A2C'
    file_postfix = f'_all'

    best_val_loss, best_val_dsc, best_val_ji = train(num_epochs, model_a2c, 
                                                train_a2c_loader, test_a2c_loader,
                                                criterion, optimizer_a2c, scheduler_a2c, 
                                                saved_dir, file_postfix, 
                                                val_every, device)

    score['a2c_loss'] = [best_val_loss]
    score['a2c_dsc'] = [best_val_dsc]
    score['a2c_ji'] = [best_val_ji]

    # Result (DataFrame)
    res = save_train_all_res_df(score)
    print(res)
        