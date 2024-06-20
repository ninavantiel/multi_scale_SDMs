import os
import torch
import wandb
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torcheval.metrics.functional import binary_auroc

from util import *
from losses import *
from models import MultimodalModel
from data.PatchesProviders import RasterPatchProvider, MultipleRasterPatchProvider, JpegPatchProvider
from data.Datasets import PatchesDatasetCooccurrences

datadir = 'data/' 
modeldir = 'models/'

def setup_model(
    env_model,
    sat_model,
    dataset,
    random_bg,
    embed_shape,
    learning_rate,
    weight_decay,
    seed,
    test=False
):
    seed_everything(seed)
    
    # make model setup dictionary
    model_setup = {}
    if env_model is not None: 
        env_model['covariates'] = [get_path_to(cov, dataset, datadir) for cov in env_model['covariates']]
        model_setup['env'] = env_model
    if sat_model is not None: 
        sat_model['covariates'] = [get_path_to(cov, dataset, datadir) for cov in sat_model['covariates']]
        model_setup['sat'] = sat_model

    assert len(model_setup) <= 2
    multimodal = (len(model_setup) == 2)
    if multimodal: assert random_bg is False
    
    if list(model_setup.values())[0]['model_name'] == 'MultiResolutionAutoencoder': 
        assert multimodal == False
        autoencoder = True
    else: 
        autoencoder = False

    if dataset == 'glc23':
        sep = ';'
        item_columns=['lat','lon','patchID','dayOfYear']
        item_columns_val=item_columns
        sat_id_col = 'patchID'
        select_sat_train=['rgb','nir']
    elif dataset == 'glc24':
        sep = ','
        item_columns=['lat','lon','surveyId','dayOfYear']
        item_columns_val=['lat','lon','surveyId']
        sat_id_col = 'surveyId'
        select_sat_train=['po_train_patches_rgb','po_train_patches_nir']
        select_sat_val=['pa_train_patches_rgb','pa_train_patches_nir']

    # covariate patch providers 
    train_providers = []
    for model_name, model_dict in model_setup.items():
        flatten = True if model_dict['model_name'] == 'MLP' else False 
        if model_name == 'env':
            train_providers.append(make_providers(model_dict['covariates'], model_dict['patch_size'], flatten))
        elif model_name == 'sat':
            train_providers.append(make_providers(model_dict['covariates'], model_dict['patch_size'], flatten, sat_id_col, select_sat_train))

    if dataset == 'glc23':
        val_providers = train_providers
    elif dataset == 'glc24':
        val_providers = []
        for model_name, model_dict in model_setup.items():
            flatten = True if model_dict['model_name'] == 'MLP' else False 
            if model_name == 'env':
                val_providers.append(make_providers(model_dict['covariates'], model_dict['patch_size'], flatten))
            elif model_name == 'sat':
                val_providers.append(make_providers(model_dict['covariates'], model_dict['patch_size'], flatten, sat_id_col, select_sat_val))

    # get paths to data for given dataset
    train_occ_path = get_path_to("po", dataset, datadir)
    if random_bg:
        random_bg_path = get_path_to("random_bg", dataset, datadir)
    else:
        random_bg_path = None
    val_occ_path = get_path_to("pa", dataset, datadir)

    # training data
    print("\nMaking dataset for training occurrences")
    train_data = PatchesDatasetCooccurrences(
        occurrences=train_occ_path, 
        providers=train_providers, 
        item_columns=item_columns,
        pseudoabsences=random_bg_path, 
        sep=sep
    )

    for i, key in enumerate(model_setup.keys()):
        model_setup[key]['input_shape'] = train_data[0][0][i].shape
        if multimodal:
            assert embed_shape is not None
            model_setup[key]['output_shape'] = embed_shape
        else:
            model_setup[key]['output_shape'] = train_data.n_species_pred
    print(f"input shape: {[params['input_shape'] for params in model_setup.values()]}")

    # validation data
    print("\nMaking dataset for validation occurrences")
    val_data = PatchesDatasetCooccurrences(
        occurrences=val_occ_path, 
        providers=val_providers, 
        item_columns=item_columns_val,
        species=train_data.species_pred, 
        sep=sep
    )

    if test:
        print("\nMaking dataset for test set")
        test_occ_path = get_path_to("test", dataset, datadir)
        test_data = PatchesDatasetCooccurrences(
            occurrences=test_occ_path, 
            providers=val_providers, 
            item_columns=item_columns_val,
            species=train_data.species_pred, 
            label_name="Id",
            sep=sep,
            test=True)
    
    # model and optimizer
    print("\nMaking model")
    model_list = [make_model(model_dict) for model_dict in model_setup.values()]
    if multimodal:
        model = MultimodalModel(
            model_list[0], model_list[1], train_data.n_species_pred, embed_shape, embed_shape
        )
    else:
        model = model_list[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if test:
        return train_data, val_data, test_data, model, optimizer, multimodal, autoencoder
    else:
        return train_data, val_data, model, optimizer, multimodal, autoencoder

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", required=True, help="Run name")
    parser.add_argument("-p", "--path_to_config", help="Path to run config file")

    args = parser.parse_args()
    run_name = args.run_name
    path_to_config = args.path_to_config

    # read config file
    if path_to_config is None:
        path_to_config = f"{modeldir}{run_name}/config.json"
    assert os.path.exists(path_to_config)
    with open(path_to_config, "r") as f:
        config = json.load(f)
    config = {k: v if v != "" else None for k,v in config.items()}

    # get variables from config 
    log_wandb = config['log_wandb']
    wandb_project = config['wandb_project']
    wandb_id = config['wandb_id']
    env_model = config['env_model']
    sat_model = config['sat_model']
    dataset = config['dataset']
    random_bg = config['random_bg']
    embed_shape = config['embed_shape']
    loss = config['loss']
    lambda2 = config['lambda2']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_workers_train = config['num_workers_train']
    num_workers_val = config['num_workers_val']
    seed = config['seed']

    # set seed
    seed_everything(seed)

    # get device (gpu (cuda) or cpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev}")

    # setup data and model
    train_data, val_data, model, optimizer, multimodal, autoencoder = setup_model(
        env_model=env_model,
        sat_model=sat_model,
        dataset=dataset,
        random_bg=random_bg,
        embed_shape=embed_shape, 
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        seed=seed) 
    model = model.to(dev)

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers_train)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=num_workers_val)

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
                'dataset': dataset, 'pseudoabsences': random_bg,
                'n_species_train': train_data.n_species_pred, 'n_species_val': val_data.n_species_pred_in_data,
                'env_model': env_model, 'sat_model': sat_model,
                'embed_shape': embed_shape, 'epochs': n_epochs, 
                'batch_size': batch_size, 'lr': learning_rate, 'weight_decay': weight_decay,
                'optimizer':'SGD', 'loss': loss, 'lambda2': lambda2, 
                # 'autoencoder_loss': autoencoder_loss_fn, 
                'val_loss': 'BCEloss', 'id': wandb_id})
        
    # load checkpoint if it exists
    if not os.path.isdir(modeldir+run_name): 
        os.mkdir(modeldir+run_name)
    if os.path.exists(f"{modeldir}{run_name}/last.pth"): 
        print(f"\nLoading model from checkpoint")
        checkpoint = torch.load(f"{modeldir}{run_name}/last.pth")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        max_val_auc = torch.load(f"{modeldir}{run_name}/best_val_auc.pth")['val_auc']
    else:
        start_epoch = 0
        max_val_auc = 0
    
    # model training
    for epoch in range(start_epoch, n_epochs):
        print(f"EPOCH {epoch}") 
        model.train()
        train_loss_list = []
        for po_inputs, bg_inputs, labels in tqdm(train_loader):
            labels = labels.to(torch.float32).to(dev) 

            # forward pass
            if multimodal:
                inputsA = po_inputs[0].to(torch.float32).to(dev)
                inputsB = po_inputs[1].to(torch.float32).to(dev)
                y_pred = torch.sigmoid(model(inputsA, inputsB))

            else:
                if not random_bg:
                    inputs = po_inputs[0].to(torch.float32).to(dev)
                else:
                    inputs = torch.cat((po_inputs[0], bg_inputs[0]), 0).to(torch.float32).to(dev)

                # if autoencoder:
                #     input_patches = [transforms.CenterCrop(size=aspp.receptive_field)(inputs) for aspp in model.aspp_branches][::-1]
                #     y_pred, xlist_decoded = model(inputs)
                #     y_pred = torch.sigmoid(y_pred)
                # else:
                y_pred = torch.sigmoid(model(inputs))
                    
            if not random_bg:
                if loss == 'weighted_loss':
                    train_loss = loss_fn(y_pred, labels, species_weights)
                else:
                    train_loss = loss_fn(y_pred, labels)
            else:
                po_pred = y_pred[0:len(po_inputs[0])]
                bg_pred = y_pred[len(po_inputs[0]):]
                if loss == 'weighted_loss':
                    train_loss = loss_fn(po_pred, labels, species_weights, lambda2, bg_pred)
                else:
                    train_loss = loss_fn(po_pred, labels, bg_pred)

            train_loss_list.append(train_loss.cpu().detach())
            
            # backward pass and weight update
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        avg_train_loss = np.mean(train_loss_list)
        print(f"{epoch}) TRAIN LOSS={avg_train_loss}")

        # evaluate model on validation set
        model.eval()
        val_loss_list, labels_list, y_pred_list = [], [], []
        for inputs, _, labels in tqdm(val_loader):
            labels = labels.to(torch.float32).to(dev) 
            labels_list.append(labels.cpu().detach().numpy())

            if multimodal:
                inputsA = inputs[0].to(torch.float32).to(dev)
                inputsB = inputs[1].to(torch.float32).to(dev)
                y_pred = torch.sigmoid(model(inputsA, inputsB))
            else:
                inputs = inputs[0].to(torch.float32).to(dev)
                y_pred = torch.sigmoid(model(inputs))

            y_pred_list.append(y_pred.cpu().detach().numpy())

            # validation loss
            val_loss = val_loss_fn(y_pred, labels).cpu().detach()
            val_loss_list.append(val_loss)
            
        avg_val_loss = np.mean(val_loss_list)
        labels = np.concatenate(labels_list)
        y_pred = np.concatenate(y_pred_list)

        # validation AUC
        labels = labels[:, val_data.species_pred_in_data]
        y_pred = y_pred[:, val_data.species_pred_in_data]            
        y_pred_torch = torch.from_numpy(y_pred.T)
        labels_torch = torch.from_numpy(labels.T)

        auc_list = binary_auroc(y_pred_torch, labels_torch, num_tasks=labels_torch.shape[0])
        auc = auc_list.median().item()
        print(f"\tVALIDATION LOSS={avg_val_loss} \nVALIDATION MEDIAN AUC={auc}")

        df = pd.DataFrame(train_data.species_counts, columns=['n_occ']).reset_index().rename(columns={'index':'species'})
        df = df[val_data.species_pred_in_data]
        df['auc'] = auc_list.cpu().detach().numpy()
        df.to_csv(f"{modeldir}{run_name}/last_species_auc.csv", index=False)

        if log_wandb:
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_auc": auc}) 

        # model checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_auc': auc
        }, f"{modeldir}/{run_name}/last.pth") 

        # save best model
        if auc > max_val_auc:
            print("Best iteration yet! Saving model...")
            max_val_auc = auc
            df.to_csv(f"{modeldir}{run_name}/best_val_auc_species_auc.csv", index=False)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_auc': auc
            }, f"{modeldir}{run_name}/best_val_auc.pth")  

# run_name = '0612_env_3'
# path_to_config = f"{modeldir}{run_name}/config.json"
# with open(path_to_config, "r") as f: 
#     config = json.load(f)
# config = {k: v if v != "" else None for k,v in config.items()}

# log_wandb = config['log_wandb']
# wandb_project = config['wandb_project']
# wandb_id = config['wandb_id']
# env_model = config['env_model']
# sat_model = config['sat_model']
# dataset = config['dataset']
# random_bg = config['random_bg']
# embed_shape = config['embed_shape']
# loss = config['loss']
# lambda2 = config['lambda2']
# n_epochs = config['n_epochs']
# batch_size = config['batch_size']
# learning_rate = config['learning_rate']
# weight_decay = config['weight_decay']
# num_workers_train = config['num_workers_train']
# num_workers_val = config['num_workers_val']
# seed = config['seed']