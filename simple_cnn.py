#import random
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import wandb

from GLC23PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider#, JpegPatchProvider
from GLC23Datasets import PatchesDataset, PatchesDatasetMultiLabel
from models import cnn

# SAMPLE DATA
# data_path = 'data/sample_data/' # root path of the data
# presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sample.csv'
# presence_absence_path = data_path+'Presence_Absences_occurrences/Presences_Absences_train_sample.csv'

# OCCURRENCE DATA
data_path = 'data/full_data/'
# presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train.csv'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sampled_100.csv'
presence_absence_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'

# COVARIATES
# p_rgb = JpegPatchProvider(data_path+'SatelliteImages/')#, dataset_stats='jpeg_patches_sample_stats.csv') # take all sentinel imagery layer (4)
# p_soil = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Soilgrids/') #9
# n_features = 15 #44 

run_name = '2604_1730_full_data_sampled_100_f1_weighted_loss'
if not os.path.exists(f"models/{run_name}"): os.makedirs(f"models/{run_name}")
n_epochs = 100
batch_size = 512
seed = 42

np.random.seed(seed) # Numpy seed also uses by Scikit Learn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dev)

    # COVARIATES
    p_bioclim = MultipleRasterPatchProvider(
        data_path+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
    ) #19
    # p_elevation = RasterPatchProvider(data_path + 'EnvironmentalRasters/Elevation/ASTER_Elevation.tif') #1
    p_hfp_d = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/detailed/') #14
    p_hfp_s = RasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif') #1

    # PRESENCE ONLY DATA: train dataset
    presence_only_df = pd.read_csv(presence_only_path, sep=";", header='infer', low_memory=False)
    train_data = PatchesDatasetMultiLabel(
        occurrences=presence_only_df.reset_index(), 
        providers=(p_bioclim, p_hfp_d, p_hfp_s)
    )
    print(f"\nTRAINING DATA: n={len(train_data)}")
    # get number of features and number of species in train dataset
    n_features = train_data[0][0].cpu().detach().shape[0]
    print(f"Number of covariates = {n_features}")
    n_species = len(train_data.unique_sorted_targets)
    print(f"Number of species = {n_species}")

    # PRESENCE ABSENCE DATA: validation dataset
    presence_absence_df = pd.read_csv(presence_absence_path, sep=";", header='infer', low_memory=False)
    val_data = PatchesDatasetMultiLabel(
        occurrences=presence_absence_df, 
        providers=(p_bioclim, p_hfp_d, p_hfp_s),
        ref_targets=train_data.unique_sorted_targets
    )
    print(f"TEST DATA: n={len(val_data)}")

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size*10, num_workers=16)

    # model
    model = cnn(n_features, n_species).to(dev)
    print(f"\nMODEL: {model}")

    # loss function
    # train_labels = torch.cat([torch.unsqueeze(train_data[i][1], dim=0) for i in range(len(train_data))])
    # weights = (train_labels.shape[0] - torch.sum(train_labels, axis=0)) / (torch.sum(train_labels, axis=0) + 1e-3)
    weights = torch.tensor((
        (len(train_data) - presence_only_df.groupby('speciesId').glcID.count()) / (presence_only_df.groupby('speciesId').glcID.count()+ 1e-3)
    ).values)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights)).to(dev)#(pred, target)
    min_loss = 99999

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    run = wandb.init(
        project='geolifeclef23', entity="deephsm", name=run_name, 
        config={'epochs': n_epochs, 'batch_size': batch_size, 'n_covariates': n_features, 'n_species': n_species}
    )

    for epoch in range(n_epochs):
        print(f"EPOCH {epoch}")

        model.train()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            # forward pass
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = loss.cpu().detach()
        print(f"{epoch}) TRAIN LOSS={train_loss}")

        # model checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
        }, f"models/{run_name}/last.pth") 
        # save best model
        if min_loss > train_loss:
            min_loss = train_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, f"models/{run_name}/best.pth")

        model.eval()
        # val_loss_list = []
        y_true_list = []
        y_pred_list = []
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            y_pred = model(inputs)
            y_true_list.append(labels)
            y_pred_list.append(y_pred)
        
        # validation loss
        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        val_loss = loss_fn(y_pred, y_true).cpu().detach()
        print(f"\tVALIDATION LOSS={val_loss}")

        # optimise threshold for f1-score
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
        thresholds = np.arange(0, 0.001, 1e-4)
        f1_scores = []
        for thresh in thresholds:
            y_bin = np.where(y_pred > thresh, 1, 0)
            f1_scores.append(np.array([f1_score(y_true[i,:], y_bin[i,:]) for i in range(y_true.shape[0])]).mean())
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)       

        # log train and val metrics for epoch
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_best_threshold": best_threshold, "val_best_f1": best_f1})        