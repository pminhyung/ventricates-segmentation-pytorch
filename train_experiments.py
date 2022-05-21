from random import shuffle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
import pandas as pd
import os
import yaml
import warnings 
warnings.filterwarnings('ignore')

from utils import set_gpu, seed_everything, load_yaml
from preprocessing import train_transform, val_transform, test_transform
from datasets import CustomDataset, make_data_df, collate_fn
from model import Encoder
from callbacks import GradualWarmupSchedulerV2, EarlyStoppingScore
from metrics import m_dsc_bin, m_ji_bin

import wandb

#wandb.config.update(args) # argparser 사용할 경우 args = parser.parse() 다음 선언

def get_config(train_dataset, val_dataset, test_dataset, optimizer, kfolds=False):
    config = {'framework': 'smp', 
        'img_size': IMGSIZE, 
        'transforms':'\n'.join(map(str, train_transform.transforms.transforms)),
        'batch_size':BATCH_SIZE,
         'epochs': num_epochs, 
         'model':f'{ENCODER_TYPE}-{DECODER_TYPE}',
         'pretrained':'True', 
         'loss':criterion.__class__.__name__,
         'optimizer': optimizer.__class__.__name__,
         'learning_rate' : PRE_ENCODER_LR,
          'weight_decay':PRE_WEIGHT_DECAY, 
          'early_stopping_patience':es_patience, 
          'early_stopping_mindelta': es_min_delta,
          'random_seed': random_seed,
          'num_train': len(train_dataset)*0.8 if kfolds else len(train_dataset),
          'num_val': len(val_dataset)*0.2 if kfolds else len(val_dataset),
          'num_test' : len(test_dataset),
          }
    return config

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

def make_and_shuffle_df(path):
    df = make_data_df(path)
    df_shuffled=df.iloc[np.random.RandomState(seed=random_seed).permutation(df.index)].reset_index(drop=True)
    return df_shuffled

def save_model(model, saved_dir, file_name):
    make_savedir(saved_dir)
    checkpoint = model.state_dict()
    ckpt_path = os.path.join(saved_dir, file_name)
    torch.save(checkpoint, ckpt_path)

def train_swa(num_epochs, model, train_loader, val_loader, swa_loader, criterion, optimizer, scheduler, saved_dir, file_postfix, val_every, device):
    print(f'Start training..')
    n_class = 2 # 0, 1 (binary)
    best_val_loss = 9999999
    early_stopping = EarlyStoppingScore(patience=es_patience, min_delta=es_min_delta)

    swa_model = AveragedModel(model)
    swa_start = 1 # GradualWarmUp의 warm up이 warmp_epo 완료 시 종료, swa_start가 끝난 이후 swa 시작
    swa_scheduler = SWALR(optimizer, swa_lr = 5e-5)

    example_ct=0

    for epoch in range(num_epochs):
        model.train()
        swa_model.train()
        for step, (images, masks, fnames) in enumerate(train_loader):
            images = torch.stack(images)       
            # masks = torch.stack(masks).long()
            masks = torch.stack(masks).float()  # (N, W, H), BCEWithLogitsLoss(target.float()필요)
            
            # # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device) # 학습시 실행
            
            # device 할당
            model = model.to(device) # 학습시 실행

            # inference
            outputs = model(images)

            # BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), masks) # (N, 1, W, H) -> (N, W, H)

            # CrossEntropyLoss
            # loss = criterion(torch.argmax(outputs, dim=1), masks) # (N,W,H); 채널 차원에서 원핫인코딩으로 변환도 시도가능

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu().numpy() # 학습시 실행
            outputs = np.where(outputs[:,0,:,:]>=.5, 1, 0) # (N, W, H)

            # logits to class (CrossEntropyLoss)
            # outputs = outputs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy() # 학습시 실행
            
            mdsc = m_dsc_bin(masks, outputs)
            mji = m_ji_bin(masks, outputs)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mDSC: {round(mdsc,4)}, mJI: {round(mji, 4)}')
            example_ct +=  len(images)
            wandb.log({ "Examples": [wandb.Image(image.permute([1,2,0]).numpy(), caption=fname) for image, fname in zip(images.detach().cpu(), fnames)],
                        "Loss" : loss.item(), "mDSC":mdsc, "mJI":mji}, step=example_ct)

        if epoch>swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(epoch)
 

        swa_model.to(device)
        torch.optim.swa_utils.update_bn(swa_loader, swa_model)
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            if epoch>swa_start:
                val_loss, vdsc, vji, mask_list = validation(epoch + 1, swa_model, val_loader, criterion, device)
            else:
                val_loss, vdsc, vji, mask_list = validation(epoch + 1, model, val_loader, criterion, device)
            wandb.log({ "predictions" : mask_list,
                        'Epoch': epoch+1,
                        "Val Loss": val_loss,
                        "val_mDSC":vdsc,
                        "val_mJI":vji,
                    # "Test Loss": test_loss,
                        })
            if val_loss < best_val_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_val_loss = val_loss
                fname = f'{ENCODER_TYPE}-{DECODER_TYPE}(pre)_ep{epoch+1}_vloss{round(val_loss.item(),4)}_vdsc{round(vdsc, 4)}_vji{round(vji,4)}'+file_postfix+'.pth'
                save_model(model, saved_dir, fname)
                wandb.alert(title="Best", text=f"dsc({round(vdsc, 4)}),ji({round(vji, 4)})",)
                
            elif vji >= 0.7: # ckpt 앙상블용
                save_model(model, saved_dir, fname)
                
            # lr_scheduler(val_loss)        
            early_stopping(vji)
            if early_stopping.early_stop:
                break
    
    best_val_loss, best_val_dsc, best_val_ji = best_val_loss.detach().cpu().item(), vdsc, vji
    wandb.alert(title="Finish", text=f"dsc({round(best_val_dsc, 4)}),ji({round(best_val_ji, 4)})",)
    return best_val_loss, best_val_dsc, best_val_ji

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, file_postfix, val_every, device):
    print(f'Start training..')
    n_class = 2 # 0, 1 (binary)
    best_val_loss = 9999999
    early_stopping = EarlyStoppingScore(patience=es_patience, min_delta=es_min_delta)
    example_ct=0

    for epoch in range(num_epochs):
        model.train() # 각 epoch 당 validation 수행 .eval()하므로 다시 .train()
        for step, (images, masks, fnames) in enumerate(train_loader):
            images = torch.stack(images)       
            # masks = torch.stack(masks).long()
            masks = torch.stack(masks).float()  # (N, W, H), BCEWithLogitsLoss(target.float()필요)
            
            # # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device) # 학습시 실행
            
            # device 할당
            model = model.to(device) # 학습시 실행

            # inference
            outputs = model(images)

            # BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), masks) # (N, 1, W, H) -> (N, W, H)

            # CrossEntropyLoss
            # loss = criterion(torch.argmax(outputs, dim=1), masks) # (N,W,H); 채널 차원에서 원핫인코딩으로 변환도 시도가능

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu().numpy() # 학습시 실행
            outputs = np.where(outputs[:,0,:,:]>=.5, 1, 0) # (N, W, H)

            # logits to class (CrossEntropyLoss)
            # outputs = outputs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy() # 학습시 실행
            
            mdsc = m_dsc_bin(masks, outputs)
            mji = m_ji_bin(masks, outputs)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mDSC: {round(mdsc,4)}, mJI: {round(mji, 4)}')
            example_ct +=  len(images)
            wandb.log({ "Examples": [wandb.Image(image.permute([1,2,0]).numpy(), caption=fname) for image, fname in zip(images.detach().cpu(), fnames)],
                        "Loss" : loss.item(), "mDSC":mdsc, "mJI":mji}, step=example_ct)

        scheduler.step(epoch)
    
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_loss, vdsc, vji, mask_list = validation(epoch + 1, model, val_loader, criterion, device)
            wandb.log({ "predictions" : mask_list,
                        'Epoch': epoch+1,
                        "Val Loss": val_loss,
                        "val_mDSC":vdsc,
                        "val_mJI":vji,
                    # "Test Loss": test_loss,
                        })
            if val_loss < best_val_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_val_loss = val_loss
                fname = f'{ENCODER_TYPE}-{DECODER_TYPE}(pre)_ep{epoch+1}_vloss{round(val_loss.item(),4)}_vdsc{round(vdsc, 4)}_vji{round(vji,4)}'+file_postfix+'.pth'
                save_model(model, saved_dir, fname)
                wandb.alert(title="Best", text=f"dsc({round(vdsc, 4)}),ji({round(vji, 4)})",)
            elif vji >= 0.7: # ckpt 앙상블용
                save_model(model, saved_dir, fname)
                
            # lr_scheduler(val_loss)        
            early_stopping(vji)
            if early_stopping.early_stop:
                break
    
    best_val_loss, best_val_dsc, best_val_ji = best_val_loss.detach().cpu().item(), vdsc, vji
    wandb.alert(title="Finish", text=f"dsc({round(best_val_dsc, 4)}),ji({round(best_val_ji, 4)})",)
    return best_val_loss, best_val_dsc, best_val_ji

segmentation_classes = ['background', 'heart']



def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()
    with torch.no_grad():
        n_class = 1
        total_loss = 0
        cnt = 0 # step cnt
        
        dsc = 0.
        ji = 0.
        data_cnt = 0 # single data cnt
        for step, (images, masks, fnames, _) in enumerate(data_loader):
            
            images = torch.stack(images)  # (N, C, W, H)
            #masks = torch.stack(masks).long()  # (N, W, H)
            masks = torch.stack(masks).float()  # (N, W, H), BCEWithLogitsLoss(target.float()필요)

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images) # (N, C, W, H)
            
            # BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), masks) # (N, 1, W, H) -> (N, W, H)

            # CrossEntropyLoss
            # loss = criterion(torch.argmax(outputs, dim=1), masks) # (N,W,H); 채널 차원에서 원핫인코딩으로 변환도 시도가능
            total_loss += loss
            cnt += 1
            
            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu().numpy() # 학습시 실행
            outputs = np.where(outputs[:,0,:,:]>=.5, 1, 0) # (N, W, H)

            # logits to class (CrossEntropyLoss)
            # outputs = outputs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy() # 학습시 실행
            
            _, sum_dsc = m_dsc_bin(masks, outputs, return_sum=True)
            _, sum_ji = m_ji_bin(masks, outputs, return_sum=True)

            dsc+=sum_dsc
            ji+=sum_ji
            data_cnt+=outputs.shape[0]
            if step==0:
                mask_list = []
                images = images.detach().cpu().numpy()

                for image, mask, path, output in list(zip(images, masks, fnames, outputs))[:num_to_log]:
                    bg_image = (image*255).transpose([1,2,0]).astype(np.uint8) # (3,W,H) -> (W,H,3)
                    # prediction_mask = output.transpose([1,2,0]).astype(np.uint8) # (1,W,H) -> (W,H,1)
                    prediction_mask = output.astype(np.uint8) # (W,H)
                    true_mask = mask.astype(np.uint8)
                    mask_list.append(wb_mask(bg_image, prediction_mask, true_mask, path))

        mdsc = dsc/data_cnt
        mji = ji/data_cnt
        avg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avg_loss.item(), 4)}, mDSC : {round(mdsc, 4)}, \
                mJI: {round(mji, 4)}')
        
    return avg_loss, mdsc, mji, mask_list

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

    IMGSIZE = config['TRAIN']['img_size']
    BATCH_SIZE = config['TRAIN']['batch_size']
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

    ENCODER_TYPE = config['TRAIN']['encoder_type']
    DECODER_TYPE = config['TRAIN']['decoder_type']
    PRETRAINED = config['TRAIN']['pretrained']

    # train_df, test_df
    train_a2c_path = './heartdata/train/A2C'
    train_a4c_path = './heartdata/train/A4C'
    val_a2c_path = './heartdata/validation/A2C'
    val_a4c_path = './heartdata/validation/A4C'

    train_a2c_df = make_data_df(train_a2c_path)
    train_a4c_df = make_data_df(train_a4c_path)
    test_a2c_df = make_data_df(val_a2c_path)
    test_a4c_df = make_data_df(val_a4c_path)

    train_a2c_df, test_a2c_df = map(make_and_shuffle_df, [train_a2c_path, val_a2c_path])
    train_a4c_df, test_a4c_df = map(make_and_shuffle_df, [train_a4c_path, val_a4c_path])

    # A2C Train DataLoader
    train_a2c_dataset = CustomDataset(df=train_a2c_df, mode='train', transform=train_transform)
    train_a2c_loader = torch.utils.data.DataLoader(
                            dataset=train_a2c_dataset, 
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            collate_fn=collate_fn,
                            drop_last=True,
                            pin_memory =True
                            #shuffle=True, # SWA 사용 시 False
                            )
    train_a2c_swaloader = torch.utils.data.DataLoader(
                            dataset=train_a2c_dataset, 
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True,
                            )

    # A4C Train DataLoader
    train_a4c_dataset = CustomDataset(df=train_a4c_df, mode='train', transform=train_transform)
    train_a4c_loader = torch.utils.data.DataLoader(
                            dataset=train_a4c_dataset, 
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            collate_fn=collate_fn,
                            drop_last=True,
                            pin_memory =True
                            #shuffle=True, # SWA 사용 시 False
                            )
    train_a4c_swaloader = torch.utils.data.DataLoader(
                            dataset=train_a4c_dataset, 
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True,
                            )

    # Test Dataset, Test DataLoader
    test_a2c_dataset = CustomDataset(df=test_a2c_df, mode='val', transform=test_transform)
    test_a2c_loader = torch.utils.data.DataLoader(
                            dataset=test_a2c_dataset, 
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            collate_fn=collate_fn,
                            pin_memory =True
                            )
    test_a4c_dataset = CustomDataset(df=test_a4c_df, mode='val', transform=test_transform)
    test_a4c_loader = torch.utils.data.DataLoader(
                            dataset=test_a4c_dataset, 
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            collate_fn=collate_fn,
                            pin_memory =True
                            )

    # Model
    model_a2c = Encoder(ENCODER_TYPE, DECODER_TYPE, pretrained=PRETRAINED)
    model_a4c = Encoder(ENCODER_TYPE, DECODER_TYPE, pretrained=PRETRAINED)

    # Optimizer
    optimizer_a2c = torch.optim.Adam(model_a2c.parameters(), lr=PRE_ENCODER_LR, weight_decay=PRE_WEIGHT_DECAY, amsgrad=False)
    optimizer_a4c = torch.optim.Adam(model_a4c.parameters(), lr=PRE_ENCODER_LR, weight_decay=PRE_WEIGHT_DECAY, amsgrad=False)

    # schedulers
    scheduler_a2c = get_scheduler(optimizer_a2c)
    scheduler_a4c = get_scheduler(optimizer_a4c)

    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        model_a2c = nn.DataParallel(model_a2c)
        model_a4c = nn.DataParallel(model_a4c)

    criterion = nn.BCEWithLogitsLoss()

    val_every = 1
    num_epochs = 1000
    num_to_log = 10
    score = {}

    ############################# A2C #############################
    saved_dir = './saved/A2C'
    file_postfix = f'_all'
    config = get_config(train_a2c_dataset, test_a2c_dataset, test_a2c_dataset, optimizer_a2c, kfolds=False)

    wandb.init(project = f"a2c", config=config, reinit=True)
    wandb.run.name = f"a2c-exp{EXP}-all"
    wandb.run.save()
    wandb.watch(model_a2c, log="all")

    # best_val_loss, best_val_dsc, best_val_ji = train(num_epochs, model_a2c, 
    #                                             train_a2c_loader, test_a2c_loader,
    #                                             criterion, optimizer_a2c, scheduler_a2c, 
    #                                             saved_dir, file_postfix, 
    #                                             val_every, device)

    best_val_loss, best_val_dsc, best_val_ji = train_swa(num_epochs, model_a2c, 
                                                train_a2c_loader, test_a2c_loader, train_a2c_swaloader,
                                                criterion, optimizer_a2c, scheduler_a2c, 
                                                saved_dir, file_postfix, 
                                                val_every, device)

    wandb.run.finish()
    score['a2c_loss'] = [best_val_loss]
    score['a2c_dsc'] = [best_val_dsc]
    score['a2c_ji'] = [best_val_ji]

    ############################# A4C #############################
    saved_dir = './saved/A4C'
    file_postfix = f'_all'
    config = get_config(train_a4c_dataset, test_a4c_dataset, test_a4c_dataset, optimizer_a4c, kfolds=False)

    wandb.init(project = f"a4c", config=config, reinit=True)
    wandb.run.name = f"a4c-exp{EXP}-all"
    wandb.run.save()
    wandb.watch(model_a4c, log="all")

    # best_val_loss, best_val_dsc, best_val_ji = train(num_epochs, model_a4c, 
    #                                             train_a4c_loader, test_a4c_loader,
    #                                             criterion, optimizer_a4c, scheduler_a4c, 
    #                                             saved_dir, file_postfix, 
    #                                             val_every, device)

    best_val_loss, best_val_dsc, best_val_ji = train_swa(num_epochs, model_a4c, 
                                                train_a4c_loader, test_a4c_loader, train_a4c_swaloader,
                                                criterion, optimizer_a4c, scheduler_a4c, 
                                                saved_dir, file_postfix, 
                                                val_every, device)

    score['a4c_loss'] = [best_val_loss]
    score['a4c_dsc'] = [best_val_dsc]
    score['a4c_ji'] = [best_val_ji]
    wandb.run.finish()

    # Result (DataFrame)
    res = save_train_all_res_df(score)
    print(res)
    