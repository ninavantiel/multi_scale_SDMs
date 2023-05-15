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
from util import seed_everything

# RUN NAME (for wandb and model directory)
# run_name = '2704_0920_full_data_sampled_100_f1_weighted_loss'
run_name = '27_04_1320_full_data_sampled_100_weighted_loss'
# run_name = '27_04_1435_full_data_sampled_100_unweighted_loss'

# HYPERPARAMETER
n_epochs = 100
batch_size = 256
seed = 42
seed_everything(seed)

# OCCURRENCE DATA
data_path = 'data/full_data/'
# presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train.csv'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sampled_100.csv'
presence_absence_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'

# COVARIATES
p_bioclim = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/') #19
# p_elevation = RasterPatchProvider(data_path + 'EnvironmentalRasters/Elevation/ASTER_Elevation.tif') #1
p_hfp_d = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/detailed/') #14
p_hfp_s = RasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif') #1
# p_rgb = JpegPatchProvider(data_path+'SatelliteImages/')#, dataset_stats='jpeg_patches_sample_stats.csv') # take all sentinel imagery layer (4)
# p_soil = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Soilgrids/') #9

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dev)
    
    # presence only data = train dataset
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

    # presence absence data = validation dataset
    presence_absence_df = pd.read_csv(presence_absence_path, sep=";", header='infer', low_memory=False)
    val_data = PatchesDatasetMultiLabel(
        occurrences=presence_absence_df, 
        providers=(p_bioclim, p_hfp_d, p_hfp_s),
        ref_targets=train_data.unique_sorted_targets
    )
    print(f"TEST DATA: n={len(val_data)}")

    # data loaders
    # train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=8)

    # model
    model = cnn(n_features, n_species).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # load best model
    checkpoint = torch.load(f"models/{run_name}/best.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # loss = checkpoint['loss']
    print(checkpoint['epoch'], checkpoint['loss'])

    # loss function
    # train_labels = torch.cat([torch.unsqueeze(train_data[i][1], dim=0) for i in range(len(train_data))])
    # weights = (train_labels.shape[0] - torch.sum(train_labels, axis=0)) / (torch.sum(train_labels, axis=0) + 1e-3)
    
    # weights = torch.tensor(((len(train_data) - presence_only_df.groupby('speciesId').glcID.count()) / 
                            # (presence_only_df.groupby('speciesId').glcID.count()+ 1e-3)).values)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights)).to(dev)#(pred, target)
    # min_loss = 99999

    y_pred_list = []
    y_true_list = []
    for inputs, labels in tqdm(val_loader):
        y_true_list.append(labels.detach().numpy())
        y_pred_list.append(model(inputs.to(dev)).detach().numpy())
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)

    np.save(f"models/{run_name}/y_pred_epoch_{str(checkpoint['epoch'])}.npy", y_pred)
    np.save(f"models/{run_name}/y_true.npy", y_true)

    # thresholds = np.arange(0, 0.5, 0.1)
    # f1_scores = []
    # for thresh in tqdm(thresholds):
    #     y_bin = np.where(y_pred > thresh, 1, 0)
    #     f1_scores.append(np.array([f1_score(y_true[i,:], y_bin[i,:]) for i in range(y_true.shape[0])]).mean())
    # best_threshold = thresholds[np.argmax(f1_scores)]
    # best_f1 = np.max(f1_scores)      
    # print(f"Best threshold={best_threshold} --> validation F1-score={best_f1}")

    # submission = pd.read_csv("data/test_blind.csv", sep=';')
    # submission_data = PatchesDataset(
    #     occurrences=submission,
    #     id_name='Id',
    #     label_name='Id',
    #     providers=(p_bioclim, p_hfp_d, p_hfp_s),
    #     ref_targets=train_data.unique_sorted_targets
    # )   
    # print(f"SUBMISSION DATA: {len(submission_data)}")