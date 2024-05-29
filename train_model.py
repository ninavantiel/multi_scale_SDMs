import os
import torch
import wandb
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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
    n_max_low_occ,
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
        n_low_occ=n_max_low_occ,
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
        n_low_occ=n_max_low_occ,
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
            n_low_occ=n_max_low_occ,
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
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=50)# lr_lambda=lr_lambda)
    
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
    # train_occ_path = eval(config['train_occ_path'])
    # random_bg_path = eval(config['random_bg_path']) if config['random_bg_path'] is not None else None
    # val_occ_path = eval(config['val_occ_path'])
    n_max_low_occ = config['n_max_low_occ']
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

    # # get paths to data for given dataset
    # train_occ_path = get_path_to("po", config['dataset'], datadir)
    # if config['random_bg']:
    #     random_bg_path = get_path_to("random_bg", config['dataset'], datadir)
    # else:
    #     random_bg_path = None
    # val_occ_path = get_path_to("pa", config['dataset'], datadir)
    
    # # make model setup dictionary
    # model_setup = {}
    # if config['env_model'] is not None: 
    #     config['env_model']['covariates'] = [get_path_to(cov, config['dataset'], datadir) for cov in config['env_model']['covariates']]
    #     model_setup['env'] = config['env_model']
    # if config['sat_model'] is not None: 
    #     config['sat_model']['covariates'] = [get_path_to(cov, config['dataset'], datadir) for cov in config['sat_model']['covariates']]
    #     model_setup['sat'] = config['sat_model']

    # set seed
    seed_everything(seed)

    # get device (gpu (cuda) or cpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev}")

    # setup data and model
    train_data, val_data, model, optimizer, multimodal, autoencoder = setup_model(
        # model_setup=model_setup, 
        env_model=env_model,
        sat_model=sat_model,
        dataset=dataset,
        random_bg=random_bg,
        # train_occ_path=train_occ_path, 
        # random_bg_path=random_bg_path, 
        # val_occ_path=val_occ_path, 
        n_max_low_occ=n_max_low_occ,
        embed_shape=embed_shape, 
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        seed=seed) 
    model = model.to(dev)

    # receptive_fields = []
    # if "env" in model_setup.keys():
    #     if model_setup["env"]["model_name"] == "MultiResolutionModel":
    #         receptive_fields = [aspp.receptive_field for aspp in model.aspp_branches]

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers_train)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=num_workers_val)

    # loss functions
    loss_fn = eval(loss)
    # if autoencoder: 
    #     autoencoder_loss_fn = nn.L1Loss()
    # else: 
    #     autoencoder_loss_fn = None
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
                # 'train_data': train_occ_path, 'pseudoabsences': random_bg_path,
                # 'val_data': val_occ_path, 
                'n_species': train_data.n_species_pred, 
                # 'n_max_low_occ': n_max_low_occ, 
                # 'n_species_low_occ': len(train_data.low_occ_species_idx),
                'env_model': env_model, 'sat_model': sat_model,
                # 'model': model_setup, #'receptive_fields': receptive_fields,
                'embed_shape': embed_shape, 'epochs': n_epochs, 
                'batch_size': batch_size, 'lr': learning_rate, 'weight_decay': weight_decay,
                'optimizer':'SGD', 'loss': loss, 'lambda2': lambda2, 
                # 'autoencoder_loss': autoencoder_loss_fn, 
                'val_loss': 'BCEloss', 'id': wandb_id
            }
        )
        
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
        print(f"EPOCH {epoch}") # (lr = {optimizer.param_groups[0]['lr']:.4f})")

        # import pdb; pdb.set_trace()
        model.train()
        train_loss_list, train_classif_loss_list, train_reconstr_loss_list = [], [], []
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
                    classif_loss = loss_fn(y_pred, labels, species_weights)
                else:
                    classif_loss = loss_fn(y_pred, labels)
            else:
                po_pred = y_pred[0:len(po_inputs[0])]
                bg_pred = y_pred[len(po_inputs[0]):]
                if loss == 'weighted_loss':
                    classif_loss = loss_fn(po_pred, labels, species_weights, lambda2, bg_pred)
                else:
                    classif_loss = loss_fn(po_pred, labels, bg_pred)
            train_loss = classif_loss
            
            # if autoencoder:
            #     reconstr_loss = [autoencoder_loss_fn(pred_patch, input_patch).cpu().detach() for pred_patch, input_patch in zip(xlist_decoded, input_patches)]
            #     train_loss = classif_loss + np.sum(reconstr_loss)
            #     train_classif_loss_list.append(classif_loss.cpu().detach())
            #     train_reconstr_loss_list.append(reconstr_loss)
            # else:
            #     train_loss = classif_loss

            train_loss_list.append(train_loss.cpu().detach())
            
            # backward pass and weight update
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # scheduler.step()
        
        avg_train_loss = np.mean(train_loss_list)
        # if autoencoder:
        #     avg_train_classif_loss = np.mean(train_classif_loss_list)
        #     avg_train_reconstr_loss = np.mean(train_reconstr_loss_list, axis=0)
        #     print(f"{epoch}) TRAIN LOSS={avg_train_loss} (classification loss={avg_train_classif_loss}, reconstruction loss={avg_train_reconstr_loss})")
        # else:
        print(f"{epoch}) TRAIN LOSS={avg_train_loss}")

        # evaluate model on validation set
        model.eval()
        val_loss_list, val_classif_loss_list, val_reconstr_loss_list, labels_list, y_pred_list = [], [], [], [], []
        for inputs, _, labels in tqdm(val_loader):
            labels = labels.to(torch.float32).to(dev) 
            labels_list.append(labels.cpu().detach().numpy())

            if multimodal:
                inputsA = inputs[0].to(torch.float32).to(dev)
                inputsB = inputs[1].to(torch.float32).to(dev)
                y_pred = torch.sigmoid(model(inputsA, inputsB))
            else:
                inputs = inputs[0].to(torch.float32).to(dev)

                # if autoencoder:
                #     input_patches = [transforms.CenterCrop(size=aspp.receptive_field)(inputs) for aspp in model.aspp_branches][::-1]
                #     y_pred, xlist_decoded = model(inputs)
                #     y_pred = torch.sigmoid(y_pred)
                # else:
                y_pred = torch.sigmoid(model(inputs))

            y_pred_list.append(y_pred.cpu().detach().numpy())

            # validation loss
            val_loss = val_loss_fn(y_pred, labels).cpu().detach()
            # if autoencoder:
            #     reconst_loss = [autoencoder_loss_fn(pred_patch, input_patch).cpu().detach() for pred_patch, input_patch in zip(xlist_decoded, input_patches)]
            #     val_reconstr_loss_list.append(reconst_loss)
            #     val_loss = classif_loss + np.sum(reconst_loss)
            # else:
            #     val_loss = classif_loss

            val_loss_list.append(val_loss)
            val_classif_loss_list.append(classif_loss)
            
        avg_val_loss = np.mean(val_loss_list)
        labels = np.concatenate(labels_list)
        y_pred = np.concatenate(y_pred_list)

        # validation AUC
        labels = labels[:, val_data.species_pred_in_data]
        y_pred = y_pred[:, val_data.species_pred_in_data]
        auc = roc_auc_score(labels, y_pred)
        print(f"\tVALIDATION LOSS={avg_val_loss} \nVALIDATION AUC={auc}")

        # validation AUC for species with low number of occurrences
        # idx_in_val_low_occ = np.logical_and(train_data.species_pred_in_low_occ, val_data.species_pred_in_data)
        # auc_low_occ = roc_auc_score(labels[:, idx_in_val_low_occ], y_pred[:, idx_in_val_low_occ])

        # validation AUC per species
        df = pd.DataFrame(train_data.species_counts, columns=['n_occ']).reset_index().rename(columns={'index':'species'})
        df['auc'] = [roc_auc_score(labels[:,i], y_pred[:,i]) for i in range(labels.shape[1])]
        df.to_csv(f"{modeldir}{run_name}/last_species_auc.csv", index=False)
            
        if epoch == 0:
            np.save(f"{modeldir}/{run_name}/val_y_true.npy", labels)
        # if autoencoder:
        #     avg_val_classif_loss = np.mean(val_classif_loss_list)
        #     avg_val_reconstr_loss = np.mean(val_reconstr_loss_list, axis=0)
        #     print(f"\tVALIDATION LOSS={avg_val_loss} (classification loss={avg_val_classif_loss}, reconstruction loss={avg_val_reconstr_loss}) \nVALIDATION AUC={auc}")
        # else:

        if log_wandb:
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_auc": auc}) #, "val_auc_low_occ": auc_low_occ})
            # if autoencoder:
            #     wandb.log({"train_total_loss": avg_train_loss, "val_total_loss": avg_val_loss})
            #     if len(avg_train_reconstr_loss) == 1:
            #         wandb.log({"train_reconstr_loss_1": avg_train_reconstr_loss[0], "val_reconstr_loss_1": avg_val_reconstr_loss[0]})
            #     elif len(avg_train_reconstr_loss) == 3:
            #         wandb.log({
            #             "train_reconstr_loss_1": avg_train_reconstr_loss[0], "train_reconstr_loss_2": avg_train_reconstr_loss[1], "train_reconstr_loss_3": avg_train_reconstr_loss[2],
            #             "val_reconstr_loss_1": avg_val_reconstr_loss[0], "val_reconstr_loss_2": avg_val_reconstr_loss[1], "val_reconstr_loss_3": avg_val_reconstr_loss[2]
            #         })
            #     else:
            #         print('cannot log reconstruction losses!')

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
            np.save(f"{modeldir}/{run_name}/best_val_auc_y_pred.npy", y_pred)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_auc': auc
            }, f"{modeldir}{run_name}/best_val_auc.pth")  

    

# run_name = '0510_multimodel_multires'
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
# n_max_low_occ = config['n_max_low_occ']
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