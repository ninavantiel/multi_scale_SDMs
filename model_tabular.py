import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import wandb

from GLC23Datasets import TabularDataset
from models import MLP
from util import seed_everything

# device (cpu ou cuda)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

# path to data
data_path = 'data/full_data/'
presence_absence_occurrences_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'
presence_absence_environment_path = data_path+'Presence_Absence_surveys/enviroTab_pa_train.csv'

# hyperparameters
batch_size = 64
learning_rate = 1e-3
num_layers = 4
width = 256
dropout = 0.5
n_epochs = 100
bin_thresh = 0.1

#covariates 
covs = [
    'x_EPSG3035', 'y_EPSG3035',
    'bio1', 'bio2', 'bio3', 'bio4','bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'bio10', 'bio11', 'bio12',
    'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19', 'landCov' 
    #'phh2o', 'sand', 'silt', 'soc', 'Built1994', 'Lights1994', 'NavWater1994', 'Built2009',
    # 'Lights2009', 'NavWater2009', 'Popdensity1990', 'Popdensity2010','Railways', 'Roads',
    # 'cec', 'cfvo', 'clay', 'nitrogen','bdod', 
]

# wandb run name
# run_name = '19_2kspecies_patch20'
run_name = '23_tabular_data_weight_mult_10_dropout_05'
print(run_name)

# seed random seed
seed = 42
seed_everything(seed)

if __name__ == "__main__":

    # presence only data = train dataset
    print("Making dataset for presence-only training data...")
    presence_absence_occurrences = pd.read_csv(presence_absence_occurrences_path, sep=";", header='infer', low_memory=False)
    presence_absence_environment = pd.read_csv(presence_absence_environment_path, sep=";", header='infer', low_memory=False)
    train_env, val_env = train_test_split(presence_absence_environment, test_size=0.1, random_state=seed)

    train_data = TabularDataset(
        occurrences=presence_absence_occurrences,
        env=train_env,
        covariates=covs,
    )
    n_covs = len(train_data.covariates)
    n_species = len(train_data.sorted_unique_targets)
    print(f"\nTRAINING DATA: n={len(train_data)}\t{n_covs} COVARIATES\t{n_species} SPECIES")

    val_data = TabularDataset(
        occurrences=presence_absence_occurrences,
        env=val_env,
        covariates=covs,
        sorted_unique_targets=train_data.sorted_unique_targets
    )
    print(f"\nVALIDATION DATA: n={len(val_data)}\t{len(val_data.covariates)} COVARIATES\t{len(val_data.sorted_unique_targets)} SPECIES")

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)#, num_workers=16)
    val_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)#, num_workers=16)

    # model and optimizer
    model =  MLP(input_size=n_covs, output_size=n_species, num_layers=num_layers, width=width, dropout=dropout).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#, momentum=0.9)

    # loss function
    # loss_fn = torch.nn.BCEWithLogitsLoss() 
    weights = torch.tensor(((len(train_data) - presence_absence_occurrences.groupby('speciesId').patchID.count()) / 
                            (presence_absence_occurrences.groupby('speciesId').patchID.count()+ 1e-3)).values * 10)
    print('Weights:', weights[0:10])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights).to(dev)

    # wandb initialization
    run = wandb.init(project='geolifeclef23', name=run_name, resume='never', config={
        'epochs': n_epochs, 'batch_size': batch_size, 'lr': learning_rate, 'n_covariates': n_covs, 'n_species': n_species, 
        'optimizer':'SGD', 'model': 'MLP', 'loss': 'BCEWithLogitsLoss', 'patch_size': 'tabular'
    })

    # # get checkpoint of model if a model has been saved
    # if not os.path.exists(f"models/{run_name}"): 
    #     os.makedirs(f"models/{run_name}")
        
    # if os.path.exists(f"models/{run_name}/last.pth"): 
    #     checkpoint = torch.load(f"models/{run_name}/last.pth")
    #     start_epoch = checkpoint['epoch'] + 1
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     min_train_loss = torch.load(f"models/{run_name}/best_train_loss.pth")['train_loss']
    #     min_val_loss = torch.load(f"models/{run_name}/best_train_loss.pth")['val_loss']
    # else:
    #     start_epoch = 0
    if not os.path.exists(f"models/{run_name}"): 
        os.makedirs(f"models/{run_name}")
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
            val_precision_list.append(precision_score(y_true.T, y_bin.T, average='macro',zero_division=0))
            val_recall_list.append(recall_score(y_true.T, y_bin.T, average='macro'))
            val_f1_list.append(f1_score(y_true.T, y_bin.T, average='macro', zero_division=0)) #, zero_division=0)
    
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

