import pandas as pd
import numpy as np
import torch 
import wandb
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score


from GLC23Datasets import RGBNIR_env_Dataset
from models import twoBranchCNN

from util import seed_everything
seed_everything(42)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

data_path = 'data/full_data/'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sampled_100.csv'
presence_absence_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'

batch_size = 64
learning_rate = 1e-3
n_epochs = 200
bin_thresh = 0.1

run_name = '24_dataset_from_scratch'
if not os.path.exists(f"models/{run_name}"): 
    os.makedirs(f"models/{run_name}")

if __name__ == "__main__":
    # split presence absence data into validation and training set
    # patches are sorted by lat/lon and then the first n_val are chosen 
    # --> train and val set are geographically separated
    presence_absence_df = pd.read_csv(presence_absence_path, sep=";", header='infer', low_memory=False)
    sorted_patches = presence_absence_df.drop_duplicates(['patchID','dayOfYear']).sort_values(['lat','lon'])
    
    n_val = round(sorted_patches.shape[0] * 0.2)
    val_patches = sorted_patches.iloc[0:n_val] 
    val_presence_absence = presence_absence_df[(presence_absence_df['patchID'].isin(val_patches['patchID'])) & 
                             (presence_absence_df['dayOfYear'].isin(val_patches['dayOfYear']))].reset_index(drop=True)
    print(f"Validation set: {n_val} patches -> {val_presence_absence.shape[0]} observations")
    train_patches = sorted_patches.iloc[n_val:]
    train_presence_absence = presence_absence_df[(presence_absence_df['patchID'].isin(train_patches['patchID'])) & 
                             (presence_absence_df['dayOfYear'].isin(train_patches['dayOfYear']))].reset_index(drop=True)
    print(f"Training set: {train_patches.shape[0]} patches -> {train_presence_absence.shape[0]} observations")

    train_dataset = RGNIR_env_Dataset(train_presence_absence, env_patch_size=10, rgbnir_patch_size=100)
    n_species = len(train_dataset.species)
    print(f"Training set: {len(train_dataset)} sites, {n_species} sites")
    val_dataset = RGNIR_env_Dataset(val_presence_absence, species=train_dataset.species, env_patch_size=10, rgbnir_patch_size=100)
    print(f"Validation set: {len(val_dataset)} sites, {len(val_dataset.species)} sites")

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8)
    
    model = twoBranchCNN(n_species).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss() 

    run = wandb.init(project='geolifeclef23', name=run_name, resume='allow', config={
        'epochs': n_epochs, 'batch_size': batch_size, 'lr': learning_rate, 'n_species': n_species, 
        'optimizer':'SGD', 'model': 'cnn_batchnorm_patchsize_20', 'loss': 'BCEWithLogitsLoss', 
        'env_patch_size': 10, 'rgb_patch_size':100, 'train_data': 'PA'
    }) #resume='never',

    if os.path.exists(f"models/{run_name}/last.pth"): 
        print(f"Loading model from checkpoint...")
        checkpoint = torch.load(f"models/{run_name}/last.pth")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        min_train_loss = torch.load(f"models/{run_name}/best_train_loss.pth")['train_loss']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, n_epochs):
        print(f"EPOCH {epoch}")

        model.train()
        train_loss_list = []
        for rgb, env, labels in tqdm(train_loader):
            y_pred = model(rgb.to(torch.float32).to(dev), env.to(torch.float32).to(dev))
            loss = loss_fn(y_pred, labels.to(torch.float32).to(dev))
            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.cpu().detach())
        avg_train_loss = np.array(train_loss_list).mean()
        print(f"\tTRAIN LOSS={avg_train_loss}")
        
        model.eval()
        val_loss_list, val_precision_list, val_recall_list, val_f1_list = [], [], [], []
        for rgb, env, labels in tqdm(val_loader):
            y_pred = model(rgb.to(torch.float32).to(dev), env.to(torch.float32).to(dev))#, val=True)
            val_loss = loss_fn(y_pred, labels.to(torch.float32).to(dev)).cpu().detach()
            val_loss_list.append(val_loss)

            y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
            y_bin = np.where(y_pred > bin_thresh, 1, 0)
            val_precision_list.append(precision_score(labels.T, y_bin.T, average='macro', zero_division=0))
            val_recall_list.append(recall_score(labels.T, y_bin.T, average='macro', zero_division=0))
            val_f1_list.append(f1_score(labels.T, y_bin.T, average='macro', zero_division=0)) 

        avg_val_loss = np.array(val_loss_list).mean()
        avg_val_precision = np.array(val_precision_list).mean()
        avg_val_recall = np.array(val_recall_list).mean()
        avg_val_f1 = np.array(val_f1_list).mean()
        print(f"\tVALIDATION LOSS={avg_val_loss}\tPRECISION={avg_val_precision}, RECALL={avg_val_recall}, F1-SCORE={avg_val_f1} (threshold={bin_thresh})")
        wandb.log({
            "train_loss": avg_train_loss, "val_loss": avg_val_loss, 
            "val_prec": avg_val_precision, "val_recall": avg_val_recall, "val_f1": avg_val_f1
        })

        # model checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, f"models/{run_name}/last.pth") 

        # save best models
        if epoch == 0: 
            min_train_loss = avg_val_loss
            
        if avg_train_loss <= min_train_loss:
            min_train_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_f1': avg_val_f1
            }, f"models/{run_name}/best_train_loss.pth")  


