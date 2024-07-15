import torch

def weighted_loss(y_pred, labels, species_weights, lambda2=1, bg_pred=None):
    '''
    Full weighted loss function from Zbinden et al. 2024.
    Modified to not include random background points
    '''
    if bg_pred is None: 
        lambda2 = 1
    batch_size = y_pred.size(0)
    loss_dl_pos = (log_loss(y_pred) * labels * species_weights.repeat((batch_size, 1))).mean()
    loss_dl_neg = (log_loss(1 - y_pred) * (1 - labels) * lambda2 * (species_weights/(species_weights - 1)).repeat((batch_size, 1))).mean() 
    if bg_pred is not None:
        loss_bg_neg = (log_loss(1 - bg_pred) * (1-lambda2)).mean()
        loss = loss_dl_pos + loss_dl_neg + loss_bg_neg
    else:
        loss = loss_dl_pos + loss_dl_neg
    return loss

def log_loss(pred):
    return -torch.log(pred + 1e-5)