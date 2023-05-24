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
n_epochs = 100
bin_thresh = 0.1

run_name = '24_dataset_from_scratch'
thresholds = np.arange(0, 0.2, 0.01)
# Best threshold=0.1 --> validation F1-score=0.04684899465936664
# Best threshold=0.05 --> validation F1-score=0.04937688956348617
# Best threshold=0.07 --> validation F1-score=0.050866337975610375

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

    # train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8)
    
    model = twoBranchCNN(n_species).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss() 

    # load best model
    print("\nLoading best train loss model checkpoint...")
    checkpoint = torch.load(f"models/{run_name}/best_train_loss.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"model checkpoint at epoch {epoch}")

    if os.path.exists(f"models/{run_name}/y_pred_epoch_{str(checkpoint['epoch'])}.npy") and os.path.exists(f"models/{run_name}/y_true.npy"):
        print("Loading y_pred and y_true...")
        y_pred =  np.load(f"models/{run_name}/y_pred_epoch_{str(checkpoint['epoch'])}.npy")
        y_true = np.load(f"models/{run_name}/y_true.npy")
    else:
        print("Computing y_pred and y_true...")
        y_pred_list = []
        y_true_list = []
        for rgb, env, labels in tqdm(val_loader):
            y_true_list.append(labels)
            batch_y_pred = torch.sigmoid(model(rgb.to(torch.float32).to(dev), env.to(torch.float32).to(dev)))
            y_pred_list.append(batch_y_pred.cpu().detach().numpy())

        y_pred = np.concatenate(y_pred_list)
        print(f"y_pred shape: {y_pred.shape}")
        np.save(f"models/{run_name}/y_pred_epoch_{str(epoch)}.npy", y_pred)
        y_true = np.concatenate(y_true_list)
        print(f"y_true shape: {y_true.shape}")
        np.save(f"models/{run_name}/y_true.npy", y_true)

    f1_scores = []
    for thresh in tqdm(thresholds):
        y_bin = np.where(y_pred > thresh, 1, 0)
        f1_scores.append(f1_score(y_true.T, y_bin.T, average='macro', zero_division=0))
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)      
    print(f"Thresholds: {thresholds}\nF1-scores: {f1_scores}")
    print(f"Best threshold={best_threshold} --> validation F1-score={best_f1}")

    print("Loading submission data...")
    submission = pd.read_csv("data/test_blind.csv", sep=';')
    submission_dataset = RGNIR_env_Dataset(submission, species=train_dataset.species, env_patch_size=10, rgbnir_patch_size=100, label_col='Id')
    print(f"SUBMISSION DATA: {len(submission_dataset)}")
    submission_loader = torch.utils.data.DataLoader(submission_dataset, shuffle=False, batch_size=batch_size, num_workers=24)

    print("Making predictions on submission data...")
    y_pred_list = []
    for rgb, env, _ in tqdm(submission_loader):
            batch_y_pred = torch.sigmoid(model(rgb.to(torch.float32).to(dev), env.to(torch.float32).to(dev)))
            y_pred_list.append(batch_y_pred.cpu().detach().numpy())

    targets = train_dataset.species
    y_pred = np.concatenate(y_pred_list)
    y_bin = np.where(y_pred > best_threshold, 1, 0)
    # np.save(f"models/{run_name}/submission_y_pred_epoch_{str(epoch)}_thresh_{str(best_threshold)}.npy", y_pred)
    # np.save(f"models/{run_name}/target.npy", targets)

    pred_species = [' '.join([str(x) for x in targets[np.where(y_pred[i, :] > best_threshold)]]) for i in range(y_pred.shape[0])]
    sub_df = pd.DataFrame({'Id': submission.Id, 'Predicted': pred_species})
    sub_df.to_csv(f"data/submissions/{run_name}_submission_epoch_{epoch}_thresh_{str(best_threshold)}.csv", index=False)