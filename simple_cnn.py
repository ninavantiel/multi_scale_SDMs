#import random
import torch
from tqdm import tqdm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import wandb

from GLC23PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider, JpegPatchProvider
from GLC23Datasets import PatchesDataset, PatchesDatasetMultiLabel
from models import cnn

# SAMPLE DATA
# data_path = 'data/sample_data/' # root path of the data
# presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sample.csv'
# presence_absence_path = data_path+'Presence_Absences_occurrences/Presences_Absences_train_sample.csv'

# OCCURRENCE DATA
data_path = 'data/full_data/'
# presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train.csv'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sampled.csv'
presence_absence_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'

# COVARIATES
# p_rgb = JpegPatchProvider(data_path+'SatelliteImages/')#, dataset_stats='jpeg_patches_sample_stats.csv') # take all sentinel imagery layer (4)
# p_soil = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Soilgrids/') #9
# n_features = 15 #44 

run_name = '2504_1049_full_data_bioclim_hfp_valistest'
if not os.path.exists(f"models/{run_name}"): os.makedirs(f"models/{run_name}")
n_epochs = 100
batch_size = 256
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

    # PRESENCE ONLY DATA
    presence_only = PatchesDatasetMultiLabel(
        occurrences=presence_only_path, 
        providers=(p_bioclim, p_hfp_d, p_hfp_s)
    )

    # get number of features and number of species in presence only dataset
    n_features = presence_only[0][0].cpu().detach().shape[0]
    print(f"Number of covariates = {n_features}")
    n_species = len(presence_only.unique_sorted_targets)
    print(f"Number of species = {n_species}")

    # split presence only dataset into train and validation sets
    train_data, val_data = train_test_split(presence_only, test_size=0.2)
    # train_data = presence_only
    print(f"\nTRAINING DATA: n={len(train_data)}")
    print(f"VALIDATION DATA: n={len(val_data)}")

    # exit()

    # PRESENCE ABSENCE DATA = test set
    test_data = PatchesDatasetMultiLabel(
        occurrences=presence_absence_path, 
        providers=(p_bioclim, p_hfp_d, p_hfp_s),
        ref_targets=presence_only.unique_sorted_targets
    )
    print(f"TEST DATA: n={len(test_data)}")

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
    # val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=4)

    # model
    model = cnn(n_features, n_species).to(dev)
    print(f"\nMODEL: {model}")

    loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight)).to(device)(pred, target)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    min_loss = 99999

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
        print(f"{epoch}) LOSS={train_loss}")

        model.eval()
        avg_aucs = []
        for inputs, labels in test_loader:
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            y_pred = model(inputs)
            # validation loss
            val_loss = loss_fn(y_pred, labels).cpu().detach()
            # validation AUC
            y_pred = y_pred.cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()
            species_idx = np.where(y_true.sum(axis=0) != 0)[0]
            auc_rocs = roc_auc_score(y_true[:,species_idx], y_pred[:,species_idx], average=None)
            avg_aucs.append(auc_rocs.mean())
        # average AUC 
        avg_auc = np.array(avg_aucs).mean() # this is not great
        print(f"{epoch}) AVG_AUC={avg_auc} (for {len(species_idx)} species)")

        # log train and val metrics for epoch
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_avg_auc": avg_auc})
        if epoch == 0: wandb.config['n_species_val'] = len(species_idx)

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
    
    # model.eval()
    # avg_aucs = []
    # for inputs, labels in test_loader:
    #     inputs = inputs.to(dev)
    #     labels = labels.to(dev)
    #     y_pred = model(inputs)
    #     # test loss
    #     test_loss = loss_fn(y_pred, labels).cpu().detach()
    #     # test AUC
    #     y_pred = y_pred.cpu().detach().numpy()
    #     y_true = labels.cpu().detach().numpy()
    #     species_idx = np.where(y_true.sum(axis=0) != 0)[0]
    #     auc_rocs = roc_auc_score(y_true[:,species_idx], y_pred[:,species_idx], average=None)
    #     avg_aucs.append(auc_rocs.mean())

    # avg_auc = np.array(avg_aucs).mean()
    # print(f"TEST SET: AVG_AUC={avg_auc} (for {len(species_idx)} species)")

    # wandb.log({"test_loss": test_loss, "test_avg_auc": avg_auc})
    # wandb.config["n_species_test"] = len(species_idx)
        