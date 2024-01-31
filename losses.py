import torch

def full_weighted_loss(y_pred, labels, species_weights):
    '''
    Full weighted loss function from Zbinden et al. 2024.
    Modified to not include random background points
    '''
    batch_size = y_pred.size(0)
    loss_dl_pos = (log_loss(y_pred) * labels * species_weights.repeat((batch_size, 1))).mean()
    loss_dl_neg = (log_loss(1 - y_pred) * (1 - labels) * (species_weights/(species_weights - 1)).repeat((batch_size, 1))).mean() 
    loss = loss_dl_pos + loss_dl_neg
    return loss

def an_slds_loss(y_pred, labels):
    site_loss = ((log_loss(y_pred) * labels) + (log_loss(1 - y_pred) * (1 - labels))).sum(axis=1)
    site_weight = 1 / (labels.sum(axis=1) + 1e-5)
    loss = (site_loss * site_weight).mean()
    return loss

def an_full_loss(y_pred, labels, lambda_=2048):
    loss_dl_pos = (log_loss(y_pred) * labels * lambda_).mean()
    loss_dl_neg = (log_loss(1 - y_pred) * (1 - labels)).mean() 
    loss = loss_dl_pos + loss_dl_neg
    return loss

def log_loss(pred):
    return -torch.log(pred + 1e-5)