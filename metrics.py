import numpy as np
# from sklearn.metrics import jaccard_score
# from scipy.spatial import distance

def m_dsc_bin(y_true, y_pred_bin, return_sum=False, return_single=False):
    """[summary]
    
    Batch에 대한 Dice Similarity Coefficient (DSC) 계산 함수 (Binary)

    return_sum : 데이터 전체의 점수 계산을 위해 합계 반환

    return_single : 추론 시 각 데이터에 대한 점수 리스트 제공

    """
    # shape of y_true and y_pred_bin: (n_samples, height, width)
    batch_size = len(y_true)
    sum_dice = 0.
    dices= []
    for i in range(batch_size):
        single_dice = single_dsc_bin(y_true[i], y_pred_bin[i])
        sum_dice += single_dice
        if return_single:
            dices.append(single_dice)
    mean_dice = sum_dice/batch_size
    if return_single:
        return dices
    if return_sum:
        return mean_dice, sum_dice
    return mean_dice

# def single_dsc_bin(y_true, y_pred_bin):
#     # shape of y_true and y_pred_bin: (height, width)
#     intersection = np.sum(y_true * y_pred_bin)
#     if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
#         return 1
#     return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

def single_dsc_bin(y_true, y_pred):
    """
    Single Data에 대한 Dice Similarity Coefficient (DSC) 계산 함수 (Binary)
    """
    # same as single_dsc_bin
    # shape of y_true and y_pred_bin: (height, width)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

def m_dsc_multichannel(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dsc_bin(y_true[i, :, :, j], y_pred_bin[i, :, :, j])
            mean_dice_channel += channel_dice/(channel_num*batch_size)
    return mean_dice_channel

def single_ji_bin(y_true, y_pred):
    """
    Single Data에 대한 Jaccard Index (JI) 계산 함수 (Binary)
    """
    # shape of y_true and y_pred: (height, width)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    dice = single_dsc_bin(y_true_f, y_pred_f)
    single_ji = dice/(2-dice)
    return single_ji

def m_ji_bin(y_true, y_pred, return_sum=False,return_single=False):
    """[summary]

    Batch에 대한 Jaccard Index (JI) 계산 함수 (Binary)

    return_sum : 데이터 전체의 점수 계산을 위해 합계 반환

    return_single : 추론 시 각 데이터에 대한 점수 리스트 제공
    
    """
        
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = len(y_true)
    sum_ji = 0.
    jis = []
    for i in range(batch_size):
        single_ji = single_ji_bin(y_true[i], y_pred[i])
        sum_ji += single_ji
        if return_single:
            jis.append(single_ji)
    mean_ji=sum_ji/batch_size
    if return_single:
        return jis
    if return_sum:
        return mean_ji, sum_ji
    return mean_ji

def iou_score(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f)
    union = np.logical_or(y_true_f, y_pred_f)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

if __name__ == '__main__':
    np.random.seed(0)
    # (batch_size, height, weight)
    true = np.random.rand(2, 3, 3)>0.5
    pred = np.random.rand(2, 3, 3)>0.5

    # # (batch_size, height, weight, channel)
    # true = np.random.rand(2, 7, 7, 3)>0.5
    # pred = np.random.rand(2, 7, 7, 3)>0.5
    
    print(m_dsc_bin(true, pred))
    print(m_ji_bin(true, pred))