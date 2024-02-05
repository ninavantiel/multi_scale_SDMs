import torch
from tqdm import tqdm
import numpy as np
import wandb
import os

from sklearn.metrics import roc_auc_score

from util import seed_everything
from losses import full_weighted_loss, an_full_loss
from data.PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider
from data.Datasets import PatchesDataset, PatchesDatasetCooccurrences
from models import MLP

# paths to data
datadir = 'data/full_data/'
po_path = datadir+'Presence_only_occurrences/Presences_only_train_sampled_100_percent_min_1_occurrences.csv'
#Presences_only_train_sampled_10_percent_min_100_occurrences.csv' #Presences_only_train_sampled_25_percent_min_10_occurrences.csv'
bg_path = datadir+'Presence_only_occurrences/Pseudoabsence_locations_bioclim_soil.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'
bioclim_dir = datadir+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
soil_dir = datadir+'EnvironmentalRasters/Soilgrids/'
landcover_dir = datadir+'EnvironmentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif'

seed = 42
seed_everything(seed)

# device (cpu ou cuda)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {dev}\n")

# hyperparameters
patch_size = 1
flatten = True
batch_size = 1024
learning_rate = 1e-3
n_epochs = 150
n_layers = 5
width = 1000
id = False
n_max_low_occ = 50
pseudoabsences = True
lambda2 = 0.5

# wandb
wandb_project = 'spatial_extent_glc23_env'
run_name = '0201_MLP_env_1x1_weighted_loss_05_all_PA_species_with_pseudoabsences'
# run_name = '0201_MLP_env_1x1_an_full_loss_all_PA_species_with_pseudoabsences'
if not os.path.isdir('models/'+run_name): 
    os.mkdir('models/'+run_name)
train_data_name = 'Presences_only_train_sampled_100_percent_min_1_occurrences'
test_data_name = 'Presences_Absences_train'
model_name = 'MLP'

if __name__ == "__main__":
    # load patch providers for covariates
    print("Making patch providers for predictor variables...")
    p_bioclim = MultipleRasterPatchProvider(bioclim_dir, size=patch_size, flatten=flatten) 
    print('biolcim done')
    p_soil = MultipleRasterPatchProvider(soil_dir, size=patch_size, flatten=flatten) 
    print('soil done')
    p_landcover = RasterPatchProvider(landcover_dir, size=patch_size, flatten=flatten)
    print('landcover done')
    ## !! LANDCOVER DOESNT HAVE SAME SCALE AS BIOCLIM AND SOIL!!
    ## !! LANCOVER: 500M, ELEVATION: 30M 
    # p_elev = RasterPatchProvider(elev_dir, size=patch_size, flatten=flatten)
    # print('elev done')

    # train data: presence only data 
    print("\nMaking dataset for presence-only training data...")
    train_data = PatchesDatasetCooccurrences(occurrences=po_path, providers=(p_bioclim, p_soil, p_landcover), pseudoabsences=bg_path)
    print(f"TRAINING DATA: n_items={len(train_data)}, n_species={len(train_data.species)}")

    n_features = train_data[0][0].shape[0]
    n_species = len(train_data.species)
    print(f"nb of features = {n_features}\nnb of species = {n_species}")

    low_occ_species = train_data.species_counts[train_data.species_counts <= n_max_low_occ].index
    low_occ_species_idx = np.where(np.isin(train_data.species, low_occ_species))[0]
    print(f"nb of species with less than {n_max_low_occ} occurrences = {len(low_occ_species_idx)}")

    # validation data: presence absence data 
    print("\nMaking dataset for presence-absence validation data...")
    val_data = PatchesDatasetCooccurrences(occurrences=pa_path, providers=(p_bioclim, p_soil, p_landcover), species=train_data.species)
    print(f"VALIDATION DATA: n_items={len(val_data)}, n_species={len(val_data.species)}")
    print(val_data[0][0].shape, val_data[0][1].shape)
    
    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=8)

    # model and optimizer
    model = MLP(n_features, n_species, n_layers, width).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss function
    loss_fn = full_weighted_loss#an_full_loss #
    species_weights = torch.tensor(train_data.species_weights).to(dev)
    val_loss_fn = torch.nn.BCEWithLogitsLoss()

    if not id: 
        id = wandb.util.generate_id() 
    print('ID: ', id)
    run = wandb.init(
        project=wandb_project, name=run_name, resume="allow", id=id,
        config={
            'epochs': n_epochs, 'batch_size': batch_size, 'lr': learning_rate, 
            'n_species': n_species, 'n_input_features': n_features,
            'train_data': train_data_name, 'test_data': test_data_name, 'pseudoabsences': pseudoabsences,
            'optimizer':'SGD', 'model': model_name, 'n_layers': n_layers, 'width': width,
            'loss': 'full_weighted_loss_w_pseudoabsences', 'val_loss': 'BCEloss', 'lambda2': lambda2,
            'env_patch_size': patch_size, 'id': id
        }
    ) 

    if os.path.exists(f"models/{run_name}/last.pth"): 
        print(f"Loading model from checkpoint...")
        checkpoint = torch.load(f"models/{run_name}/last.pth")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        min_val_loss = torch.load(f"models/{run_name}/best_val_loss.pth")['val_loss']
    else:
        start_epoch = 0

    # model training
    for epoch in range(start_epoch, n_epochs):
        print(f"EPOCH {epoch}")

        model.train()
        train_loss_list = []
        for batch in tqdm(train_loader):
            if not pseudoabsences:
                inputs, labels = batch 
                inputs = inputs.to(torch.float32).to(dev)
                labels = labels.to(torch.float32).to(dev) 

                # forward pass
                y_pred = model(inputs)
                y_pred_sigmoid = torch.sigmoid(y_pred)
                loss = loss_fn(y_pred_sigmoid, labels, species_weights)
            
            else:
                inputs, labels, bg_inputs = batch
                concat_inputs = torch.cat((inputs, bg_inputs), 0).to(torch.float32).to(dev)
                labels = labels.to(torch.float32).to(dev) 

                # forward pass
                output = model(concat_inputs)
                y_pred = output[0:len(inputs)]
                bg_pred = output[len(inputs):]
                y_pred_sigmoid = torch.sigmoid(y_pred)
                bg_pred_sigmoid = torch.sigmoid(bg_pred)
                loss = loss_fn(y_pred_sigmoid, labels, species_weights, lambda2, bg_pred_sigmoid)
            
            train_loss_list.append(loss.cpu().detach())

            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = np.array(train_loss_list).mean()
        print(f"{epoch}) TRAIN LOSS={avg_train_loss}")

        model.eval()
        val_loss_list, val_train_loss_list, labels_list, y_pred_list = [], [], [], []
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 
            labels_list.append(labels.cpu().detach().numpy())

            y_pred = model(inputs)
            y_pred_sigmoid = torch.sigmoid(y_pred)
            y_pred_list.append(y_pred_sigmoid.cpu().detach().numpy())

            # validation loss
            val_loss = val_loss_fn(y_pred, labels) 
            val_loss_list.append(val_loss.cpu().detach())
            
            val_train_loss = loss_fn(y_pred_sigmoid, labels, species_weights)
            val_train_loss_list.append(val_train_loss.cpu().detach())
    
        avg_val_loss = np.array(val_loss_list).mean()
        avg_val_train_loss = np.array(val_train_loss_list).mean()
        labels = np.concatenate(labels_list)
        y_pred = np.concatenate(y_pred_list)
        auc = roc_auc_score(labels, y_pred)
        auc_low_occ = roc_auc_score(labels[:, low_occ_species_idx], y_pred[:, low_occ_species_idx])
        print(f"\tVALIDATION LOSS={avg_val_loss} \tVALIDATION AUC={auc}")

        wandb.log({
            "train_loss": avg_train_loss, "val_loss": avg_val_loss, 
            "val_train_loss": avg_val_train_loss, 
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
        if epoch == 0: 
            min_val_loss = avg_val_loss
            
        if avg_val_loss <= min_val_loss:
            min_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_auc': auc
            }, f"models/{run_name}/best_val_loss.pth")  

    # # final model evaluation
    # print('\nFINAL MODEL EVALUATION')
    # checkpoint = torch.load(f"models/{run_name}/best_val_loss.pth")
    # best_val_loss_epoch = checkpoint['epoch']

    # model = MLP(n_features, n_species, n_layers, width).to(dev)
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # model.eval()
    # val_loss_list, labels_list, y_pred_list = [], [], []

    # for inputs, labels in tqdm(val_loader):
    #     inputs = inputs.to(torch.float32).to(dev)
    #     labels = labels.to(torch.float32).to(dev)
    #     labels_list.append(labels.cpu().detach().numpy())

    #     y_pred = model(inputs)
    #     y_pred_sigmoid = torch.sigmoid(y_pred)
    #     y_pred_list.append(y_pred_sigmoid.cpu().detach().numpy())

    #     val_loss = loss_fn(y_pred, labels)
    #     val_loss_list.append(val_loss.cpu().detach())    

    # labels = np.concatenate(labels_list)
    # y_pred = np.concatenate(y_pred_list)
    # auc = roc_auc_score(labels, y_pred)
    # wandb.log({"best_val_loss_epoch": best_val_loss_epoch, "best_val_loss_auc": auc})

