import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import wandb

from GLC23PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider, JpegPatchProvider
from GLC23Datasets import PatchesDataset, PatchesDatasetMultiLabel
from models import cnn, cnn_batchnorm, cnn_batchnorm_act, cnn_batchnorm_patchsize_20
from util import seed_everything

# device (cpu ou cuda)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

# path to data
data_path = 'data/full_data/'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sampled_100.csv'
# presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sampled_1000.csv'
# presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train.csv'
presence_absence_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'

# hyperparameters
batch_size = 64
learning_rate = 1e-3
n_epochs = 100#20
bin_thresh = 0.1
patch_size = 20#128

# wandb run name
# run_name = '15_full_data_sampled_100_less_covs'
# run_name = '16_full_data_sampled_100_batchnorm_2fclayers'
# run_name = '16_full_data_sampled_100_noact_last_layer'
# run_name = '17_bcewithlogitsloss_2'
# run_name = '18_bcewithlogitsloss_weighted'
# run_name = '19_bcewithlogitsloss_weighted_lr1e-3' # full dataset
# run_name = '19_sampled_data_patch_size_20'
# run_name = '22_sampled1000_patch_size_20_weighted_loss'
# run_name = '23_rgb_not_norm'
run_name = '23_rgb_not_norm_patchsize_20'
print(run_name)

# seed random seed
seed = 42
seed_everything(seed)

if __name__ == "__main__":
    # load patch providers for covariates
    print("Making patch providers for predictor variables...")
    p_bioclim = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/', size=patch_size, normalize=True) 
    # p_elevation = RasterPatchProvider(data_path + 'EnvironmentalRasters/Elevation/ASTER_Elevation.tif') 
    # p_hfp_d = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/detailed/') 
    # p_hfp_s = RasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif', size=patch_size, normalize=True)
    p_rgb = JpegPatchProvider(data_path+'SatelliteImages/', normalize=False, size=patch_size)
    # p_soil = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Soilgrids/', size=patch_size, normalize=True)
    # p_landcover = RasterPatchProvider(data_path+'EnvironmentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif', size=patch_size, normalize=True)
    
    # presence only data = train dataset
    print("Making dataset for presence-only training data...")
    presence_only_df = pd.read_csv(presence_only_path, sep=";", header='infer', low_memory=False)
    train_data = PatchesDatasetMultiLabel(
        occurrences=presence_only_df.reset_index(), 
        providers=(p_bioclim, p_rgb) #p_bioclim, p_hfp_s, p_soil, p_landcover, 
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
        providers=(p_bioclim, p_rgb), #p_bioclim, p_hfp_s, p_soil, p_landcover, 
        sorted_unique_targets=train_data.sorted_unique_targets
    )
    print(f"VALIDATION DATA: n={len(val_data)}")

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=8)

    # model and optimizer
    model = cnn_batchnorm_patchsize_20(n_features, n_species).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#, momentum=0.9)

    # loss function
    # loss_fn = torch.nn.BCEWithLogitsLoss() 
    weights = torch.tensor(((len(train_data) - presence_only_df.groupby('speciesId').glcID.count()) / 
                            (presence_only_df.groupby('speciesId').glcID.count()+ 1e-3)).values)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights)).to(dev)#(pred, target)

    # wandb initialization
    run = wandb.init(project='geolifeclef23', name=run_name, resume='never', config={
        'epochs': n_epochs, 'batch_size': batch_size, 'lr': learning_rate, 'n_covariates': n_features, 'n_species': n_species, 
        'optimizer':'SGD', 'model': 'cnn_batchnorm_patchsize_20', 'loss': 'BCEWithLogitsLoss', 'patch_size': patch_size
    })

    # get checkpoint of model if a model has been saved
    if not os.path.exists(f"models/{run_name}"): 
        os.makedirs(f"models/{run_name}")
        
    if os.path.exists(f"models/{run_name}/last.pth"): 
        checkpoint = torch.load(f"models/{run_name}/last.pth")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        min_train_loss = torch.load(f"models/{run_name}/best_train_loss.pth")['train_loss']
        min_val_loss = torch.load(f"models/{run_name}/best_train_loss.pth")['val_loss']
    else:
        start_epoch = 0

    # model training
    for epoch in range(start_epoch, n_epochs):
        print(f"EPOCH {epoch} (lr={learning_rate})")

        model.train()
        train_loss_list = []
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev)
            # forward pass
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.cpu().detach())

        avg_train_loss = np.array(train_loss_list).mean()
        print(f"{epoch}) TRAIN LOSS={avg_train_loss}")

        model.eval()
        val_loss_list, val_precision_list, val_recall_list, val_f1_list = [], [], [], []
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev)
            y_pred = model(inputs)
            # validation loss
            val_loss = loss_fn(y_pred, labels).cpu().detach()
            val_loss_list.append(val_loss)

            y_true = labels.cpu().detach().numpy()
            y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
            y_bin = np.where(y_pred > bin_thresh, 1, 0)
            val_precision_list.append(precision_score(y_true.T, y_bin.T, average='macro'))
            val_recall_list.append(recall_score(y_true.T, y_bin.T, average='macro'))
            val_f1_list.append(f1_score(y_true.T, y_bin.T, average='macro')) #, zero_division=0)
    
        avg_val_loss = np.array(val_loss_list).mean()
        avg_val_precision = np.array(val_precision_list).mean()
        avg_val_recall = np.array(val_recall_list).mean()
        avg_val_f1 = np.array(val_f1_list).mean()
        print(f"\tVALIDATION LOSS={avg_val_loss}, PRECISION={avg_val_precision}, RECALL={avg_val_recall}, F1-SCORE={avg_val_f1} (threshold={bin_thresh})")

        # log train and val metrics for epoch
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_prec": avg_val_precision, 
                   "val_recall": avg_val_recall, "val_f1": avg_val_f1})

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
            min_train_loss = avg_train_loss
            min_val_loss = avg_val_loss
            
        if avg_train_loss <= min_train_loss:
            min_train_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_f1': avg_val_f1
            }, f"models/{run_name}/best_train_loss.pth")  

        if avg_val_loss <= min_val_loss:
            min_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_f1': avg_val_f1
            }, f"models/{run_name}/best_val_loss.pth")  

