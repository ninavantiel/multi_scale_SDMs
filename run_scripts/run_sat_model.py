import torch
from tqdm import tqdm
import numpy as np
import wandb
import os

from sklearn.metrics import roc_auc_score

from util import seed_everything
from losses import full_weighted_loss, an_slds_loss, an_full_loss
from data.PatchesProviders import JpegPatchProvider
from data.Datasets import PatchesDatasetCooccurrences
from models import get_resnet

# paths to data
datadir = 'data/full_data/'
po_path = datadir+'Presence_only_occurrences/Presences_only_train_sampled_10_percent_min_100_occurrences.csv' #Presences_only_train_sampled_25_percent_min_10_occurrences.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'
sat_dir = datadir+'SatelliteImages/'

seed = 42
seed_everything(seed)

# device (cpu ou cuda)


# hyperparameters
patch_size = 1
flatten = True
batch_size = 1024
learning_rate = 1e-3
n_epochs = 500
id = False 

# wandb
wandb_project = 'spatial_extent_glc23_sat'
run_name = '0130_resnet_sat_128'
if not os.path.isdir('models/'+run_name): 
    os.mkdir('models/'+run_name)
train_data_name = 'Presences_only_train_sampled_10_percent_min_100_occurrences'
test_data_name = 'Presences_Absences_train'
model_name = 'ResNet18'

if __name__ == "__main__":
    # load patch providers for covariates
    print("Making patch providers for predictor variables...")
    p_sat = JpegPatchProvider(sat_dir, size=128)

    # train data: presence only data 
    print("Making dataset for presence-only training data...")
    train_data = PatchesDatasetCooccurrences(occurrences=po_path, providers=(p_sat))
    print(f"\nTRAINING DATA: n_items={len(train_data)}, n_species={len(train_data.species)}")
    print(train_data[0][0].shape, train_data[0][1].shape)

    n_features = train_data[0][0].shape[0]
    n_species = len(train_data.species)
    print(f"nb of features = {n_features}")

    # validation data: presence absence data 
    print("Making dataset for presence-absence validation data...")
    val_data = PatchesDatasetCooccurrences(occurrences=pa_path, providers=(p_sat), species=train_data.species)
    print(f"\nVALIDATION DATA: n_items={len(val_data)}, n_species={len(val_data.species)}")
    print(val_data[0][0].shape, val_data[0][1].shape)
    
    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=8)

    # model and optimizer
    model = get_resnet(n_features, n_species).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss function
    loss_fn = full_weighted_loss#an_full_loss###an_slds_loss#
    species_weights = torch.tensor(train_data.species_weights).to(dev)
    # bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    if not id: 
        id = wandb.util.generate_id() 
    print('ID: ', id)
    run = wandb.init(
        project=wandb_project, name=run_name, resume="allow", id=id,
        config={
            'epochs': n_epochs, 'batch_size': batch_size, 'lr': learning_rate, 
            'n_species': n_species, 'n_input_features': n_features,
            'train_data': train_data_name, 'test_data': test_data_name,
            'optimizer':'SGD', 'model': model_name, 
            'loss': 'full_weighted_loss', 'sat_patch_size': patch_size, 'id': id
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
        for inputs, labels in tqdm(train_loader):

            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 

            # forward pass
            y_pred = model(inputs)
            y_pred_sigmoid = torch.sigmoid(y_pred)
            loss = loss_fn(y_pred_sigmoid, labels, species_weights)
            train_loss_list.append(loss.cpu().detach())

            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = np.array(train_loss_list).mean()
        print(f"{epoch}) TRAIN LOSS={avg_train_loss}")

        model.eval()
        val_loss_list, val_bce_loss_list, labels_list, y_pred_list = [], [], [], []
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 
            labels_list.append(labels.cpu().detach().numpy())

            y_pred = model(inputs)
            y_pred_sigmoid = torch.sigmoid(y_pred)
            y_pred_list.append(y_pred_sigmoid.cpu().detach().numpy())

            # validation loss
            val_loss = loss_fn(y_pred_sigmoid, labels, species_weights)
            val_loss_list.append(val_loss.cpu().detach())

            # val_bce_loss = bce_loss_fn(y_pred, labels)
            # val_bce_loss_list.append(val_bce_loss.cpu().detach())
    
        avg_val_loss = np.array(val_loss_list).mean()
        # avg_val_bce_loss = np.array(val_bce_loss_list).mean()
        labels = np.concatenate(labels_list)
        y_pred = np.concatenate(y_pred_list)
        auc = roc_auc_score(labels, y_pred)
        print(f"\tVALIDATION LOSS={avg_val_loss} \tVALIDATION AUC={auc}")

        wandb.log({
            "train_loss": avg_train_loss, "val_loss": avg_val_loss, 
            # "val_bce_loss": avg_val_bce_loss, 
            "val_auc": auc
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

