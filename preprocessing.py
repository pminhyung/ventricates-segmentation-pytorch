import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_yaml

# # Config Parsing
# CONFIG_PATH = './config.yaml'
# config = load_yaml(CONFIG_PATH)
# IMGSIZE = config['TRAIN']['img_size']

train_transform = lambda size: A.Compose([
                            A.Resize(size, size, always_apply=True), 
                            A.pytorch.ToTensorV2()
                            ])

val_transform = lambda size: A.Compose([
                          A.Resize(size, size, always_apply=True), 
                          A.pytorch.ToTensorV2()
                          ])

test_transform = lambda size: A.Compose([
                            A.Resize(size, size, always_apply=True), 
                           A.pytorch.ToTensorV2()
                           ])