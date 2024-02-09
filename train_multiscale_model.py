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
from data.Datasets import MultiScalePatchesDatasetCooccurrences, PatchesDatasetCooccurrences
from models import MLP, ShallowCNN, get_resnet, MultiScaleModel
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

def train_multiscale_model(
    run_name, 
    log_wandb, 
    wandb_project,
    wandb_id=None, 
    train_occ_path=po_path, 
    val_occ_path=pa_path, 
    n_max_low_occ=50,
    patch_sizes=[], 
    covariates=[], # list of path to covariate directories or tif files
    model_names=['MLP','ResNet'], #'MLP', 'CNN', 'ResNet'
    embed_shape=512,
    n_layers=5, 
    width=1000, 
    n_conv_layers=2,
    n_filters=[32, 64],
    kernel_size=3, 
    pooling_size=1, 
    dropout=0.0,
    loss='weighted_loss', #'an_full_loss'
    lambda2=1,
    n_epochs=150, 
    batch_size=128, 
    learning_rate=1e-3, 
    seed=42
):
    seed_everything(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev}")

    # covariate patch providers 
    flatten = [True if model == 'MLP' else False for model in model_names]
    print(f"\nMaking patch providers for covariates: patch sizes={patch_sizes}, flatten={flatten}")
    providers = []
    for group, patch_size, f in zip(covariates, patch_sizes, flatten):

        group_providers = []
        for cov in group:
            print(f"\t - {cov}")
            if 'SatelliteImages' in cov:
                if f and patch_size != 1: 
                    exit("jpeg patch provider for satellite images cannot flatten image patches")
                group_providers.append(JpegPatchProvider(cov, size=patch_size))
            elif '.tif' in cov:
                group_providers.append(RasterPatchProvider(cov, size=patch_size, flatten=f))
            else:
                group_providers.append(MultipleRasterPatchProvider(cov, size=patch_size, flatten=f))

        providers.append(group_providers)

    # training data
    print("\nMaking dataset for training occurrences")
    train_data = MultiScalePatchesDatasetCooccurrences(occurrences=train_occ_path, providersA=providers[0], providersB=providers[1])
    n_species = train_data.n_species

    input_shapes = [train_data[0][0].shape, train_data[0][1].shape]
    print(f"input shape = {input_shapes}")

    low_occ_species = train_data.species_counts[train_data.species_counts <= n_max_low_occ].index
    low_occ_species_idx = np.where(np.isin(train_data.species, low_occ_species))[0]
    print(f"nb of species with less than {n_max_low_occ} occurrences = {len(low_occ_species_idx)}")

    # validation data
    print("\nMaking dataset for validation occurrences")
    val_data = MultiScalePatchesDatasetCooccurrences(occurrences=val_occ_path, providersA=providers[0], providersB=providers[1], species=train_data.species)
    
    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)#, num_workers=4)

    # model 
    models = []
    for model, input_shape in zip(model_names, input_shapes):
        print(model, input_shape)
        if model == 'MLP':
            model = MLP(input_shape[0], embed_shape, 
                    n_layers, width, dropout).to(dev)
        elif model == 'CNN':
            model = ShallowCNN(input_shape[0], patch_size, embed_shape,
                               n_conv_layers, n_filters, width, 
                               kernel_size, pooling_size, dropout).to(dev)
        elif model == 'ResNet':
            if input_shape[0] != 3 and input_shape[0] != 4:
                exit("ResNet adapted only for 3 or 4 input bands")
            model = get_resnet(embed_shape, input_shape[0])
        models.append(model)
        
    model = MultiScaleModel(
        models[0], models[1], n_species, embed_shape, embed_shape
    ).to(dev)

    # optimizer
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
                'n_species': n_species, 'n_input_features': input_shapes, 
                'epochs': n_epochs, 'batch_size': batch_size, 'lr': learning_rate, 
                'optimizer':'SGD', 'model': 'MultiScaleModel', 
                'loss': loss, 'val_loss': 'BCEloss', 'lambda2': lambda2,
                'patch_sizes': patch_sizes, 'id': id
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
        for inputsA, inputsB, labels in tqdm(train_loader):
            inputsA = inputsA.to(torch.float32).to(dev)
            inputsB = inputsB.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 

            # forward pass
            y_pred = torch.sigmoid(model(inputsA, inputsB))
            if loss == 'weighted_loss':
                train_loss = loss_fn(y_pred, labels, species_weights)
            else:
                train_loss = loss_fn(y_pred, labels)
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
        for inputsA, inputsB, labels in tqdm(val_loader):
            inputsA = inputsA.to(torch.float32).to(dev)
            inputsB = inputsB.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 
            labels_list.append(labels.cpu().detach().numpy())

            y_pred = torch.sigmoid(model(inputsA, inputsB))
            y_pred_list.append(y_pred.cpu().detach().numpy())

            # validation loss
            val_loss = val_loss_fn(y_pred, labels) 
            val_loss_list.append(val_loss.cpu().detach())
            
        avg_val_loss = np.array(val_loss_list).mean()
        labels = np.concatenate(labels_list)
        y_pred = np.concatenate(y_pred_list)
        auc = roc_auc_score(labels, y_pred)
        auc_low_occ = roc_auc_score(labels[:, low_occ_species_idx], y_pred[:, low_occ_species_idx])
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

if __name__ == "__main__": 
    train_multiscale_model(
        run_name='0208_multiscale_test',
        log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
        train_occ_path=po_path_sampled_25,
        val_occ_path=pa_path,
        patch_sizes=[1, 128],
        covariates=[[bioclim_dir], [sat_dir]],
        batch_size=64
    )