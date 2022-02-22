# Heart Datathon 2021 
## 주제 1 심초음파 데이터셋을 활용한 좌심실 분할 AI모델 개발 
  
### 팀명: 마633 (박민형)  
### 대회 기간 : 21.11.26 ~ 21.12.07  
  
### 최종모델:
- A2C : Efficientnet-b6 + Unet++ (checkpoint ensemble) (size 768)
    - timm-efficientnet-b6-UnetPlusPlus(pre)_ep17_vloss0.0198_vdsc0.9449_vji0.8977_all.pth
    - timm-efficientnet-b6-UnetPlusPlus(pre)_ep18_vloss0.019_vdsc0.9483_vji0.9039_all.pth
    
- A4C : Efficientnet-b5 + Unet++ (size 512)
    - timm-efficientnet-b5-UnetPlusPlus(pre)_ep16_vloss0.0159_vdsc0.964_vji0.9312_all.pth

### 실행방법:

'Directory 구조' 도표처럼 'heartdata'라는 데이터 폴더를 위치한 후, 아래 순서대로 Terminal 에서 안내된 명령문을 실행합니다

- Directory 구조 확인 및 데이터 폴더 위치
- Conda 가상환경 생성
- 학습 (Train)
- 검증 (Validation)
- 테스트 (Test)

### Directory 구조
*** 데이터 디렉토리는 아래 'heartdata'와 같이 명명 및 위치해주시면 됩니다  

`checkpoints` : 학습된 모델 checkpoint 저장 directory  
`heartdata` : Train(A2C, A4C), Validation(A2C, A4C), Test(A2C, A4C) 폴더 포함하는 데이터 Directory  
`val_pred` : Validation set 예측 Mask 저장 Directory
`test_pred` : Test set 예측 Masks 저장 Directory

```
.
├── m633_environment.yml
|
├── checkpoints
│   ├── timm-efficientnet-b6-UnetPlusPlus(pre)_ep17_vloss0.0198_vdsc0.9449_vji0.8977_all.pth (A2C)
│   ├── timm-efficientnet-b6-UnetPlusPlus(pre)_ep18_vloss0.019_vdsc0.9483_vji0.9039_all.pth (A2C)
│   └── timm-efficientnet-b5-UnetPlusPlus(pre)_ep16_vloss0.0159_vdsc0.964_vji0.9312_all.pth (A4C)
│
├── heartdata
│   ├── train
│   |   ├── A2C
|   |   └── A4C
│   ├── validation
│   |   ├── A2C
|   |   └── A4C
│   └── test
│       ├── A2C
|       └── A4C
│
├── val_pred 
├── test_pred 
│
├── train.py
├── validation.py
├── test.py
├── callbacks.py
├── datasets.py
├── loss.py
├── metrics.py
├── model.py
├── preprocesing.py
├── utils.py
└── requirements.txt
```


### Conda 가상환경 생성
```
conda create --name m633 python=3.7
conda env create --file m633_environment.yml
conda activate m633
```

### 학습 (Train) : 
- Train set을 학습 및 학습모델 저장 (제출모델 재현가능)
```
python a2c_train.py
python a4c_train.py
```

### 검증 (Validation) : 
- Validation Set에 대한 Predictions 생성 (val_pred 디렉토리에 저장)
- 생성된 Predictions와 정답을 통해 DSC, JI Score 출력
```
python a2c_validation.py
python a4c_validation.py
```

### 테스트 (Test) : 
- Test Set에 대한 Predictions 생성 (test_pred 디렉토리에 저장)
```
python a2c_test.py
python a4c_test.py
```
