import wandb

def labels(segmentation_classes):
  l = {}
  for i, label in enumerate(segmentation_classes):
    l[i] = label
  return l

def wb_mask(bg_img, pred_mask, true_mask, path):
  return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask,  "class_labels" : labels()},
    "ground truth" : {"mask_data" : true_mask, "class_labels" : labels()}}, 
    caption=path)