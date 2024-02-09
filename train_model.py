import os
import time
import torch
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from util import seed_everything
from data.PatchesProviders import RasterPatchProvider, MultipleRasterPatchProvider, JpegPatchProvider
from data.Datasets import PatchesDatasetCooccurrences
from models import MLP, ShallowCNN, get_resnet
from losses import weighted_loss, an_full_loss

datadir = 'data/full_data/'

po_path = datadir+'Presence_only_occurrences/Presences_only_train_sampled_100_percent_min_1_occurrences.csv'
po_path_sampled_25 = datadir+'Presence_only_occurrences/Presences_only_train_sampled_25_percent_min_1_occurrences.csv'
bg_path = datadir+'Presence_only_occurrences/Pseudoabsence_locations_bioclim_soil.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

sat_dir = datadir+'SatelliteImages/'
bioclim_dir = datadir+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
soil_dir = datadir+'EnvironmentalRasters/Soilgrids/'
human_footprint_path = datadir+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif'
landcover_path = datadir+'EnvironmentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif'
# elevation_path = datadir+'EnvironmentalRasters/Elevation/ASTER_Elevation.tif'

def make_providers(covariate_paths_list, patch_size, flatten):
    print(f"\nMaking patch providers for covariates: size={patch_size}x{patch_size}, flatten={flatten}")
    providers = []
    for cov in covariate_paths_list:
        print(f"\t - {cov}")
        if 'SatelliteImages' in cov:
            if flatten and patch_size != 1: 
                print("jpeg patch provider for satellite images cannot flatten image patches")
                return
            providers.append(JpegPatchProvider(cov, size=patch_size))
        elif '.tif' in cov:
            providers.append(RasterPatchProvider(cov, size=patch_size, flatten=flatten))
        else:
            providers.append(MultipleRasterPatchProvider(cov, size=patch_size, flatten=flatten))
    return providers

def get_model(model_name, n_features, n_species, model_params=None, patch_size=None):
    if model_name == 'MLP':
        model = MLP(n_features, n_species, 
                    model_params['n_layers'], 
                    model_params['width'], 
                    model_params['dropout'])
    elif model_name == 'CNN':
        model = ShallowCNN(n_features, patch_size, n_species,
                           model_params['n_conv_layers'], 
                           model_params['n_filters'], 
                           model_params['width'], 
                           model_params['kernel_size'], 
                           model_params['pooling_size'], 
                           model_params['dropout'])
    elif model_name == 'ResNet':
        if n_features != 3 and n_features != 4:
            exit("ResNet adapted only for 3 or 4 input bands")
        model = get_resnet(n_species, n_features)
    return model


def train_model(
    run_name, 
    log_wandb, 
    wandb_project,
    wandb_id=None, 
    train_occ_path=po_path, 
    random_bg_path=None, 
    val_occ_path=pa_path, 
    # n_max_low_occ=50,
    patch_size=1, 
    covariates = [], # path to covariate directories or tif files
    model='MLP', #choices: 'MLP', 'CNN', 'ResNet'
    model_params={},
    # n_layers=5, 
    # width=1000, 
    # n_conv_layers=2,
    # n_filters=[32, 64],
    # kernel_size=3, 
    # pooling_size=1, 
    # dropout=0.0,
    loss='weighted_loss', #choices: 'weighted_loss', 'an_full_loss'
    lambda2=1,
    n_epochs=150, 
    batch_size=1024, 
    learning_rate=1e-3, 
    seed=42
):
    seed_everything(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev}")

    # covariate patch providers 
    flatten = True if model == 'MLP' else False
    providers = make_providers(covariates, patch_size, flatten)

    # training data
    train_data = PatchesDatasetCooccurrences(occurrences=train_occ_path, providers=providers, pseudoabsences=random_bg_path)
    n_species = train_data.n_species
    input_shape = train_data[0][0].shape
    print(f"input shape = {input_shape}")

    # validation data
    print("\nMaking dataset for validation occurrences")
    val_data = PatchesDatasetCooccurrences(occurrences=val_occ_path, providers=providers, species=train_data.species)
    
    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)#, num_workers=4)

    # model and optimizer
    model = get_model(model, input_shape[0], n_species, model_params, patch_size).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss functions
    loss_fn = eval(loss)
    species_weights = torch.tensor(train_data.species_weights).to(dev)
    val_loss_fn = torch.nn.BCELoss()

    # log run in wandb
    if log_wandb:
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
        print(f"\nwandb id: {wandb_id}")
        run = wandb.init(
            project=wandb_project, name=run_name, resume="allow", id=wandb_id,
            config={
                'n_species': n_species, 'n_input_features': input_shape[0], 'pseudoabsences': random_bg,
                'epochs': n_epochs, 'batch_size': batch_size, 'lr': learning_rate, 
                'optimizer':'SGD', 'model': model, 'n_layers': n_layers, 'width': width,
                'loss': loss, 'val_loss': 'BCEloss', 'lambda2': lambda2,
                'patch_size': patch_size, 'id': wandb_id
            }
        )
    
    # load checkpoint if it exists
    if not os.path.isdir('models/'+run_name): 
        os.mkdir('models/'+run_name)
    if os.path.exists(f"models/{run_name}/last.pth"): 
        print(f"\nLoading model from checkpoint")
        checkpoint = torch.load(f"models/{run_name}/last.pth")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        max_val_auc = torch.load(f"models/{run_name}/best_val_auc.pth")['val_auc']
    else:
        start_epoch = 0
        max_val_auc = 0

     # model training
    for epoch in range(start_epoch, n_epochs):
        print(f"EPOCH {epoch}")

        model.train()
        train_loss_list = []
        for po_inputs, labels, bg_inputs in tqdm(train_loader):
            if random_bg_path is None:
                inputs = po_inputs.to(torch.float32).to(dev)
            else:
                inputs = torch.cat((inputs, bg_inputs), 0).to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 
            
            # forward pass
            y_pred = torch.sigmoid(model(inputs))
            if random_bg_path is None:
                train_loss = loss_fn(y_pred, labels, species_weights)
            else:
                bg_pred = y_pred[len(po_inputs):]
                y_pred = y_pred[0:len(po_inputs)]
                train_loss = loss_fn(y_pred, labels, species_weights, lambda2, bg_pred)
            train_loss_list.append(train_loss.cpu().detach())

            # backward pass and weight update
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        avg_train_loss = np.array(train_loss_list).mean()
        print(f"{epoch}) TRAIN LOSS={avg_train_loss}")

        # evaluate model on validation set
        model.eval()
        val_loss_list, labels_list, y_pred_list = [], [], []
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 
            labels_list.append(labels.cpu().detach().numpy())

            y_pred = torch.sigmoid(model(inputs))
            y_pred_list.append(y_pred.cpu().detach().numpy())

            # validation loss
            val_loss = val_loss_fn(y_pred, labels) 
            val_loss_list.append(val_loss.cpu().detach())
            
        avg_val_loss = np.array(val_loss_list).mean()
        labels = np.concatenate(labels_list)
        y_pred = np.concatenate(y_pred_list)
        auc = roc_auc_score(labels, y_pred)
        auc_low_occ = roc_auc_score(labels[:, train_data.low_occ_species_idx], y_pred[:, train_data.low_occ_species_idx])
        print(f"\tVALIDATION LOSS={avg_val_loss} \tVALIDATION AUC={auc}")

        if log_wandb:
            wandb.log({
                "train_loss": avg_train_loss, "val_loss": avg_val_loss, 
                "val_auc": auc, "val_auc_low_occ": auc_low_occ
            })

        # model checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_auc': auc
        }, f"models/{run_name}/last.pth") 

        # save best models 
        if auc > max_val_auc:
            max_val_auc = auc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_auc': auc
            }, f"models/{run_name}/best_val_auc.pth")  
