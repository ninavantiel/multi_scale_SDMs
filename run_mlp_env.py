import torch
from tqdm import tqdm
import numpy as np
import wandb
import os

from data.PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider
from data.Datasets import PatchesDataset
from models import MLP

# paths to data
datadir = 'data/full_data/'
bioclim_dir = datadir+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
soil_dir = datadir+'EnvironmentalRasters/Soilgrids/'
po_path = datadir+'Presence_only_occurrences/Presences_only_train_sampled_10_percent_min_100_occurrences.csv' #Presences_only_train_sampled_25_percent_min_10_occurrences.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

# device (cpu ou cuda)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {dev}\n")

# hyperparameters
patch_size = 16
flatten = True
batch_size = 1024
learning_rate = 1e-3
n_epochs = 150
n_layers = 5
width = 1000
id = False #'fa05ftlf'

# wandb
run_name = '0117_MLP_env_16x16_flat_train_tinyPO'
if not os.path.isdir('models/'+run_name): os.mkdir('models/'+run_name)
train_data_name = 'Presences_only_train_sampled_10_percent_min_100_occurrences'
test_data_name = 'Presences_Absences_train'
model_name = 'MLP'

if __name__ == "__main__":
    # load patch providers for covariates
    print("Making patch providers for predictor variables...")
    p_bioclim = MultipleRasterPatchProvider(bioclim_dir, size=patch_size, flatten=flatten) # size=1)
    p_soil = MultipleRasterPatchProvider(soil_dir, size=patch_size, flatten=flatten) # size=1)

    # train data: presence only data 
    print("Making dataset for presence-only training data...")
    train_data = PatchesDataset(occurrences=po_path, providers=(p_bioclim, p_soil))
    print(f"\nTRAINING DATA: n_items={len(train_data)}, n_species={len(train_data.species)}")
    print(train_data[0][0].shape, train_data[0][1].shape)

    n_features = train_data[0][0].shape[0]
    n_species = len(train_data.species)
    print(f"nb of features = {n_features}")

    # validation data: presence absence data 
    print("Making dataset for presence-absence validation data...")
    val_data = PatchesDataset(occurrences=pa_path, providers=(p_bioclim, p_soil), species=train_data.species)
    print(f"\nVALIDATION DATA: n_items={len(val_data)}, n_species={len(val_data.species)}")
    print(val_data[0][0].shape, val_data[0][1].shape)
    
    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=8)

    # model and optimizer
    model = MLP(n_features, n_species, n_layers, width).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss function
    loss_fn = torch.nn.BCEWithLogitsLoss().to(dev) # add weights to loss?

    if not id: 
        id = wandb.util.generate_id() 
    print('ID: ', id)
    run = wandb.init(
        project='spatial_extent_glc23', name=run_name, resume="allow", id=id,
        config={
            'epochs': n_epochs, 'batch_size': batch_size, 'lr': learning_rate, 
            'n_species': n_species, 'n_input_features': n_features,
            'train_data': train_data_name, 'test_data': test_data_name,
            'optimizer':'SGD', 'model': model_name, 'n_layers': n_layers, 'width': width,
            'loss': 'BCEWithLogitsLoss', 'env_patch_size': 1, 'id': id
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
            loss = loss_fn(y_pred, labels)
            train_loss_list.append(loss.cpu().detach())

            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = np.array(train_loss_list).mean()
        print(f"{epoch}) TRAIN LOSS={avg_train_loss}")

        model.eval()
        val_loss_list = []
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 
            y_pred = model(inputs)

            # validation loss
            val_loss = loss_fn(y_pred, labels)
            val_loss_list.append(val_loss.cpu().detach())

            # y_true = labels.cpu().detach().numpy()
            # y_pred =  y_pred.cpu().detach().numpy()
            # y_bin = np.where(y_pred > 0.5, 1, 0)
            # f1 = f1_score(y_true, y_bin, average='macro', zero_division=0)
            # val_f1_list.append(f1)
    
        avg_val_loss = np.array(val_loss_list).mean()
        # avg_val_f1 = np.array(val_f1_list).mean()
        print(f"\tVALIDATION LOSS={avg_val_loss}")#, F1-SCORE={avg_val_f1} (threshold=0.5)")

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})

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
            min_val_loss = avg_val_loss
            
        if avg_val_loss <= min_val_loss:
            min_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, f"models/{run_name}/best_val_loss.pth")  

