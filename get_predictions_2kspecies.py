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
from models import cnn, cnn_batchnorm, cnn_batchnorm_patchsize_20
from util import seed_everything

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# device (cpu ou cuda)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

# path to data
data_path = 'data/full_data/'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_2k_species.csv'
presence_absence_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv' # Presences_Absences_train_sampled.csv

# hyperparameters
batch_size = 64
learning_rate = 1e-2
patch_size = 20

# wandb run name
# run_name = '19_2kspecies_patch20'
# run_name = '22_2kspecies_lr1e-2'
run_name = '22_2kspecies_lr1e-2_weights_mult_10'
print(run_name)

# seed random seed
seed = 42
seed_everything(seed)

# thresholds to test
thresholds = np.arange(0, 1, 0.1)

if __name__ == "__main__":
    # load patch providers for covariates
    print("Making patch providers for predictor variables...")
    p_bioclim = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/', size=patch_size, normalize=True) 
    # p_elevation = RasterPatchProvider(data_path + 'EnvironmentalRasters/Elevation/ASTER_Elevation.tif') 
    # p_hfp_d = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/detailed/') 
    p_hfp_s = RasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif', size=patch_size, normalize=True)
    # p_rgb = JpegPatchProvider(data_path+'SatelliteImages/', size=patch_size)#, dataset_stats='jpeg_patches_sample_stats.csv') # take all sentinel imagery layer (4)
    p_soil = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Soilgrids/', size=patch_size, normalize=True)
    p_landcover = RasterPatchProvider(data_path+'EnvironmentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif', size=patch_size, normalize=True)
    
    # presence only data = train dataset
    print("Making dataset for presence-only training data...")
    presence_only_df = pd.read_csv(presence_only_path, sep=";", header='infer', low_memory=False)
    train_data = PatchesDatasetMultiLabel(
        occurrences=presence_only_df.reset_index(), 
        providers=(p_bioclim, p_hfp_s, p_soil, p_landcover)
    )
    print(f"\nTRAINING DATA: n={len(train_data)}")

    # get number of features and number of species in train dataset
    n_features = train_data[0][0].cpu().detach().shape[0]
    print(f"Number of covariates = {n_features}")
    n_species = len(train_data.sorted_unique_targets)
    print(f"Number of species = {n_species}")

    # presence absence data = validation dataset
    print("Making dataset for presence-absence validation data...")
    presence_absence_df = pd.read_csv(presence_absence_path, sep=";", header='infer', low_memory=False)
    val_data = PatchesDatasetMultiLabel(
        occurrences=presence_absence_df, 
        providers=(p_bioclim, p_hfp_s, p_soil, p_landcover),
        sorted_unique_targets=train_data.sorted_unique_targets
    )
    print(f"VALIDATION DATA: n={len(val_data)}")

    # data loaders
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=16)

    # model
    model = cnn_batchnorm_patchsize_20(n_features, n_species).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#, momentum=0.9)

    # load best model
    print("\nLoading best train loss model checkpoint...")
    checkpoint = torch.load(f"models/{run_name}/best_train_loss.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"model checkpoint at epoch {epoch}")

    # load best model
    # print("\nLoading best validation loss model checkpoint...")
    # checkpoint = torch.load(f"models/{run_name}/best_val_loss.pth")
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # print(f"model checkpoint at epoch {epoch}")

    if os.path.exists(f"models/{run_name}/y_pred_epoch_{str(checkpoint['epoch'])}.npy") and os.path.exists(f"models/{run_name}/y_true.npy"):
        print("Loading y_pred and y_true...")
        y_pred =  np.load(f"models/{run_name}/y_pred_epoch_{str(checkpoint['epoch'])}.npy")
        y_true = np.load(f"models/{run_name}/y_true.npy")
    else:
        print("Computing y_pred and y_true...")
        y_pred_list = []
        y_true_list = []
        for inputs, labels in tqdm(val_loader):
            y_true_list.append(labels)
            batch_y_pred = model(inputs.to(dev))
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
        f1_scores.append(np.array([f1_score(y_true[i,:], y_bin[i,:]) for i in range(y_true.shape[0])]).mean())
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)      
    print(f"Thresholds: {thresholds}\nF1-scores: {f1_scores}")
    print(f"Best threshold={best_threshold} --> validation F1-score={best_f1}")

    print("Loading submission data...")
    submission = pd.read_csv("data/test_blind.csv", sep=';')
    submission_data = PatchesDataset(
        occurrences=submission,
        id_name='Id',
        label_name='Id',
        providers=(p_bioclim, p_hfp_s, p_soil, p_landcover)
        # sorted_unique_targets=train_data.sorted_unique_targets
    )   
    print(f"SUBMISSION DATA: {len(submission_data)}")
    submission_loader = torch.utils.data.DataLoader(submission_data, shuffle=False, batch_size=batch_size, num_workers=24)

    print("Making predictions on submission data...")
    y_pred_list = []
    for inputs, labels in tqdm(submission_loader):
        inputs = inputs.to(dev)
        batch_y_pred = model(inputs)
        y_pred_list.append(batch_y_pred.cpu().detach().numpy())

    targets = train_data.sorted_unique_targets
    y_pred = np.concatenate(y_pred_list)
    y_bin = np.where(y_pred > best_threshold, 1, 0)
    np.save(f"models/{run_name}/submission_y_pred_epoch_{str(epoch)}_thresh_{str(best_threshold)}.npy", y_pred)
    np.save(f"models/{run_name}/target.npy", targets)

    pred_species = [' '.join([str(x) for x in targets[np.where(y_pred[i, :] > best_threshold)]]) for i in range(y_pred.shape[0])]
    sub_df = pd.DataFrame({'Id': submission.Id, 'Predicted': pred_species})
    sub_df.to_csv(f"data/submissions/{run_name}_submission_epoch_{epoch}_thresh_{str(best_threshold)}.csv", index=False)