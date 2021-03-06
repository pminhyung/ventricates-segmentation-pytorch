import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
import wandb
import os
from argparse import ArgumentParser
import warnings 
warnings.filterwarnings('ignore')

from utils import set_gpu, seed_everything, load_yaml, make_dir, make_savedir
from preprocessing import train_transform, val_transform, test_transform
from datasets import CustomDataset, make_data_df, collate_fn
from model import Encoder
from callbacks import GradualWarmupSchedulerV2, EarlyStoppingScore
from metrics import m_dsc_bin, m_ji_bin
from mywb import labels, wb_mask

# Fix Seed
SEED = 21 # 2021
seed_everything(SEED) 

# Choose GPUs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NUM = torch.cuda.device_count()

if GPU_NUM:
    GPUS = [_ for _ in range(GPU_NUM)]
    set_gpu(GPUS)

def set_global_train_vars(args):

    # Config Parsing
    config_path = f"./{args.task}_config.yaml"
    config = load_yaml(config_path)

    # Set Global Variables (Hyperparameters)
    globals().update(
                    dict(
                        IMG_SIZE = config['PREPROCESSING']['img_size'],

                        PRE_COSINE_EPO = config['PREPROCESSING']['cosine_epo'],
                        PRE_WARMUP_EPO = config['PREPROCESSING']['warmup_epo'],
                        PRE_FREEZE_EPO = config['PREPROCESSING']['freeze_epo'],
                        PRE_WARMUP_FACTOR = config['PREPROCESSING']['warmup_factor'],
                        PRE_WEIGHT_DECAY = config['PREPROCESSING']['weight_decay'],
                        PRE_ENCODER_LR = config['PREPROCESSING']['encoder_lr'],
                        PRE_EPOCHS = config['PREPROCESSING']['cosine_epo'] + config['PREPROCESSING']['warmup_epo'] + config['PREPROCESSING']['freeze_epo'],
                        PRE_SCHEDULER = config['PREPROCESSING']['scheduler'],

                        BATCH_SIZE = config['TRAIN']['batch_size'],
                        NUM_EPOCHS = config['TRAIN']['num_epochs'],
                        SEGMENTATION_CLASSES = config['TRAIN']['segmentation_classes'],
                        ES_PATIENCE = config['TRAIN']['es_patience'],
                        ES_MIN_DELTA = config['TRAIN']['es_min_delta'],
                        VAL_EVERY = config['TRAIN']['val_every'],
                        NUM_EPOCHS = config['TRAIN']['num_epochs'],
                        NUM_TO_LOG = config['TRAIN']['num_to_log'],
                        ENCODER_TYPE = config['TRAIN']['encoder_type'],
                        DECODER_TYPE = config['TRAIN']['decoder_type'],
                        PRETRAINED = True
                        )
                    )

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

def save_model(model, saved_dir, file_name):
    make_savedir(saved_dir)
    checkpoint = model.state_dict()
    ckpt_path = os.path.join(saved_dir, file_name)
    torch.save(checkpoint, ckpt_path)

def get_wb_config(train_dataset, val_dataset, test_dataset, criterion, optimizer, kfolds=False):
    config = {'framework': 'smp', 
            'img_size': IMGSIZE, 
            'transforms':'\n'.join(map(str, train_transform.transforms.transforms)),
            'batch_size':BATCH_SIZE,
            'epochs': NUM_EPOCHS, 
            'model':f'{ENCODER_TYPE}-{DECODER_TYPE}',
            'pretrained':'True', 
            'loss': criterion.__class__.__name__,
            'optimizer': optimizer.__class__.__name__,
            'learning_rate' : PRE_ENCODER_LR,
            'weight_decay':PRE_WEIGHT_DECAY, 
            'early_stopping_patience': ES_PATIENCE, 
            'early_stopping_mindelta': ES_MIN_DELTA,
            'random_seed': SEED,
            'num_train': len(train_dataset)*0.8 if kfolds else len(train_dataset),
            'num_val': len(val_dataset)*0.2 if kfolds else len(val_dataset),
            'num_test' : len(test_dataset),
            }
    return config

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, file_postfix, val_every, DEVICE):
    print(f'Start training..')
    best_val_loss = 9999999
    early_stopping = EarlyStoppingScore(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)

    example_ct = 0

    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks, fnames) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).float()  # (N, W, H), BCEWithLogitsLoss(target.float()??????)
            
            # # gpu ????????? ?????? DEVICE ??????
            images, masks = images.to(DEVICE), masks.to(DEVICE) # ????????? ??????
            
            # DEVICE ??????
            model = model.to(DEVICE) # ????????? ??????
            
            # inference
            outputs = model(images)

            # BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), masks) # (N, 1, W, H) -> (N, W, H)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu().numpy() # ????????? ??????
            outputs = np.where(outputs[:,0,:,:]>=.5, 1, 0) # (N, W, H)

            masks = masks.detach().cpu().numpy() # ????????? ??????
            
            mdsc = m_dsc_bin(masks, outputs)
            mji = m_ji_bin(masks, outputs)

            # step ????????? ?????? loss ??????
            if (step + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mDSC: {round(mdsc,4)}, mJI: {round(mji, 4)}')
            example_ct +=  len(images)
            wandb.log({ "Examples": [wandb.Image(image.permute([1,2,0]).numpy(), caption=fname) for image, fname in zip(images.detach().cpu(), fnames)],
                        "Loss" : loss.item(), "mDSC":mdsc, "mJI":mji}, step=example_ct)
            
        scheduler.step(epoch)
        
        # validation ????????? ?????? loss ?????? ??? best model ??????
        if (epoch + 1) % val_every == 0:
            val_loss, vdsc, vji, mask_list = validation(epoch + 1, model, val_loader, criterion, DEVICE)
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

            elif vji >= 0.7: # ckpt ????????????
                save_model(model, saved_dir, fname)
                     
            early_stopping(vji)
            if early_stopping.early_stop:
                break
    
    best_val_loss, best_val_dsc, best_val_ji = best_val_loss.detach().cpu().item(), vdsc, vji
    wandb.alert(title="Finish", text=f"dsc({round(best_val_dsc, 4)}),ji({round(best_val_ji, 4)})",)
    return best_val_loss, best_val_dsc, best_val_ji

def validation(epoch, model, data_loader, criterion, DEVICE):
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
            masks = torch.stack(masks).float()  # (N, W, H), BCEWithLogitsLoss(target.float()??????)

            images, masks = images.to(DEVICE), masks.to(DEVICE)            
            
            # DEVICE ??????
            model = model.to(DEVICE)
            
            outputs = model(images) # (N, C, W, H)
            
            # BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), masks) # (N, 1, W, H) -> (N, W, H)
            
            total_loss += loss
            cnt += 1
            
            # logits to class (BCEWithLogitsLoss)
            outputs = outputs.detach().cpu().numpy() # ????????? ??????
            outputs = np.where(outputs[:,0,:,:]>=.5, 1, 0) # (N, W, H)

            masks = masks.detach().cpu().numpy() # ????????? ??????
            
            _, sum_dsc = m_dsc_bin(masks, outputs, return_sum=True)
            _, sum_ji = m_ji_bin(masks, outputs, return_sum=True)

            dsc+=sum_dsc
            ji+=sum_ji
            data_cnt+=outputs.shape[0]
            if step==0:
                mask_list = []
                images = images.detach().cpu().numpy()

                for image, mask, path, output in list(zip(images, masks, fnames, outputs))[:NUM_TO_LOG]:
                    bg_image = (image*255).transpose([1,2,0]).astype(np.uint8) # (3,W,H) -> (W,H,3)
                    prediction_mask = output.astype(np.uint8) # (W,H)
                    true_mask = mask.astype(np.uint8)
                    mask_list.append(wb_mask(bg_image, prediction_mask, true_mask, path))

        mdsc = dsc/data_cnt
        mji = ji/data_cnt
        avg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avg_loss.item(), 4)}, mDSC : {round(mdsc, 4)}, \
                mJI: {round(mji, 4)}')
        
    return avg_loss, mdsc, mji, mask_list

def save_train_all_res_df(args, score):
    df = pd.DataFrame(score).T.reset_index()
    save_dir = './trainall'
    make_dir(save_dir)
    df.to_csv(f'./trainall/{args.task}-all-res.csv', index=False)
    return df

def main(args):

    set_global_train_vars(args)

     # Dataset, Dataloader 
    train_path = f'./heartdata/train/{args.task.upper()}'
    val_path = f'./heartdata/validation/{args.task.upper()}'

    train_df = make_data_df(train_path)
    test_df = make_data_df(val_path)

    train_dataset = CustomDataset(df=train_df, mode='train', transform=train_transform(IMG_SIZE))
    train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset, 
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            collate_fn=collate_fn,
                            drop_last=True
                            )

    test_dataset = CustomDataset(df=test_df, mode='val', transform=test_transform(IMG_SIZE))
    test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset, 
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            collate_fn=collate_fn,
                            )

    # Model
    model = Encoder(ENCODER_TYPE, DECODER_TYPE, pretrained=PRETRAINED)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr = PRE_ENCODER_LR, 
                                    weight_decay = PRE_WEIGHT_DECAY, 
                                    amsgrad = False)

    # schedulers
    scheduler = get_scheduler(optimizer)

    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()

    score = {}
    saved_dir = f'./saved/{args.task}'
    file_postfix = f'_all'

    config = get_wb_config(train_dataset, test_dataset, test_dataset, criterion, optimizer, kfolds=False)
    wandb.init(project = f"HDAI2021", config=config, reinit=True) # HeartDatathonAI 2021
    wandb.run.name = f"{args.task}" # a2c or a4c
    wandb.run.save()
    wandb.watch(model, log="all")

    best_val_loss, best_val_dsc, best_val_ji = train(NUM_EPOCHS, model, 
                                                train_loader, test_loader,
                                                criterion, optimizer, scheduler, 
                                                saved_dir, file_postfix, 
                                                VAL_EVERY, DEVICE)
    
    score[f'{args.task}_loss'] = [best_val_loss]
    score[f'{args.task}_dsc'] = [best_val_dsc]
    score[f'{args.task}_ji'] = [best_val_ji]

    # Result (DataFrame)
    res = save_train_all_res_df(args, score)
    print(res)

    wandb.run.finish()
    return

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--task', type=str, choices = ['a2c', 'a4c'])
    args = parser.parse_args()

    main(args)
        