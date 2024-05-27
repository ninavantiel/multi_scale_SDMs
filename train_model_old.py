import os
import torch
import wandb
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from util import seed_everything
from data.PatchesProviders import RasterPatchProvider, MultipleRasterPatchProvider, JpegPatchProvider
from data.Datasets import PatchesDatasetCooccurrences
from models import *
from losses import *

datadir = '/scratch/datasets/GeoLifeCLEF2023/' #'data/full_data/'
modeldir = '/scratch/eceo-students/vantiel/glc23/models/' # 'models/'

po_path = datadir+'Presence_only_occurrences/Presences_only_train_sampled_100_percent_min_1_occurrences.csv' #PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv
po_path_sampled_25 = datadir+'Presence_only_occurrences/Presences_only_train_sampled_25_percent_min_1_occurrences.csv'
po_path_sampled_50 = datadir+'Presence_only_occurrences/Presences_only_train_sampled_50_percent_min_0_occurrences.csv'
bg_path = datadir+'Presence_only_occurrences/Pseudoabsence_locations_bioclim_soil.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

sat_dir = datadir+'SatelliteImages/'
bioclim_dir = datadir+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
soil_dir = datadir+'EnvironmentalRasters/Soilgrids/'
human_footprint_path = datadir+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif'
landcover_path = datadir+'EnvironmentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif'

def make_providers(covariate_paths_list, patch_size, flatten):
    print(f"\nMaking patch providers with size={patch_size}x{patch_size}, flatten={flatten} for covariates:")
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

def make_model(model_dict):
    assert {'input_shape', 'output_shape'}.issubset(set(model_dict.keys()))

    if model_dict['model_name'] == 'MLP':
        param_names = {'n_layers', 'width', 'dropout'}
        assert param_names.issubset(set(model_dict.keys()))

        model = MLP(model_dict['input_shape'][0],
                    model_dict['output_shape'], 
                    model_dict['n_layers'], 
                    model_dict['width'], 
                    model_dict['dropout'])
        
    elif model_dict['model_name'] == 'CNN':
        param_names = {
            'patch_size', 'n_conv_layers', 'n_filters', 'width', 'kernel_size', 
            'padding', 'pooling_size', 'dropout', 'pool_only_last'
        }
        assert param_names.issubset(set(model_dict.keys()))
        assert model_dict['n_conv_layers'] == len(model_dict['n_filters'])

        model = ShallowCNN(model_dict['input_shape'][0],
                           model_dict['patch_size'], 
                           model_dict['output_shape'],
                           model_dict['n_conv_layers'], 
                           model_dict['n_filters'], 
                           model_dict['width'], 
                           model_dict['kernel_size'], 
                           model_dict['padding'],
                           model_dict['pooling_size'], 
                           model_dict['dropout'],
                           model_dict['pool_only_last'])
        
    elif model_dict['model_name'] == 'ResNet':
        assert 'pretrained' in list(model_dict.keys())

        model = get_resnet(
            model_dict['output_shape'], 
            model_dict['input_shape'][0], 
            model_dict['pretrained'])
        
    elif model_dict['model_name'] in ['MultiResolutionModel', 'MultiResolutionAutoencoder']:
        param_names = {'backbone', 'patch_size', 'backbone_params', 'aspp_params'}
        assert param_names.issubset(set(model_dict.keys()))

        assert model_dict['backbone'] in ['CNN', 'ResNet']
        if model_dict['backbone'] == 'CNN':
            backbone_param_names = {'n_filters', 'kernel_sizes', 'paddings', 'pooling_sizes'}
        elif model_dict['backbone'] == 'ResNet':
            backbone_param_names = {'pretrained'}
        assert backbone_param_names.issubset(set(model_dict['backbone_params'].keys()))

        aspp_param_names = {'out_channels', 'out_size', 'kernel_sizes', 'dilations', 'pooling_sizes', 'n_linear_layers'}
        assert aspp_param_names.issubset(set(model_dict['aspp_params'].keys()))

        if model_dict['model_name'] == 'MultiResolutionModel':
            model = MultiResolutionModel(
                model_dict['input_shape'][0],
                model_dict['patch_size'],
                model_dict['output_shape'],
                model_dict['backbone'],
                model_dict['backbone_params'], 
                model_dict['aspp_params'])
            
        elif model_dict['model_name'] == 'MultiResolutionAutoencoder':
            model = MultiResolutionAutoencoder(
                model_dict['input_shape'][0],
                model_dict['patch_size'],
                model_dict['output_shape'],
                model_dict['backbone'],
                model_dict['backbone_params'], 
                model_dict['aspp_params'])

    return model

def setup_model(
    model_setup,
    train_occ_path=po_path,
    random_bg_path=None,
    val_occ_path=pa_path,
    n_max_low_occ=50,
    embed_shape=None,
    learning_rate=1e-3,
    weight_decay=0,
    seed=42
):
    seed_everything(seed)
    assert len(model_setup) <= 2
    multimodal = (len(model_setup) == 2)
    if multimodal: assert random_bg_path is None
    
    if list(model_setup.values())[0]['model_name'] == 'MultiResolutionAutoencoder': 
        assert multimodal == False
        autoencoder = True
    else: 
        autoencoder = False

    # covariate patch providers 
    providers = []
    for model_dict in model_setup.values():
        flatten = True if model_dict['model_name'] == 'MLP' else False 
        providers.append(make_providers(
            model_dict['covariates'], model_dict['patch_size'], flatten
        ))

    # training data
    print("\nMaking dataset for training occurrences")
    train_data = PatchesDatasetCooccurrences(
        occurrences=train_occ_path, 
        providers=providers, 
        pseudoabsences=random_bg_path, 
        n_low_occ=n_max_low_occ
    )

    for i, key in enumerate(model_setup.keys()):
        model_setup[key]['input_shape'] = train_data[0][0][i].shape
        if multimodal:
            assert embed_shape is not None
            model_setup[key]['output_shape'] = embed_shape
        else:
            model_setup[key]['output_shape'] = train_data.n_species
    print(f"input shape: {[params['input_shape'] for params in model_setup.values()]}")

    # validation data
    print("\nMaking dataset for validation occurrences")
    val_data = PatchesDatasetCooccurrences(
        occurrences=val_occ_path, 
        providers=providers, 
        species=train_data.species, 
        n_low_occ=n_max_low_occ
    )
    
    # model and optimizer
    model_list = [make_model(model_dict) for model_dict in model_setup.values()]
    if multimodal:
        model = MultimodalModel(
            model_list[0], model_list[1], train_data.n_species, embed_shape, embed_shape
        )
    else:
        model = model_list[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=50)# lr_lambda=lr_lambda)

    return train_data, val_data, model, optimizer, multimodal, autoencoder

def train_model(
    run_name, 
    log_wandb, 
    model_setup,
    wandb_project=None,
    wandb_id=None, 
    train_occ_path=po_path, 
    random_bg_path=None, 
    val_occ_path=pa_path, 
    n_max_low_occ=50,
    embed_shape=None,
    loss='weighted_loss', 
    lambda2=1,
    n_epochs=150, 
    batch_size=128, 
    learning_rate=1e-3, 
    weight_decay=0,
    num_workers_train=8,
    num_workers_val=4,
    seed=42
):
    seed_everything(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev}")

    train_data, val_data, model, optimizer, multimodal, autoencoder = setup_model(
        model_setup=model_setup, 
        train_occ_path=train_occ_path, 
        random_bg_path=random_bg_path, 
        val_occ_path=val_occ_path, 
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
    val_loss_fn = nn.BCELoss()

    # log run in wandb
    if log_wandb:
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
        print(f"\nwandb id: {wandb_id}")
        run = wandb.init(
            project=wandb_project, name=run_name, resume="allow", id=wandb_id,
            config={
                'train_data': train_occ_path, 'pseudoabsences': random_bg_path,
                'val_data': val_occ_path, 'n_species': train_data.n_species, 
                'n_max_low_occ': n_max_low_occ, 
                'n_species_low_occ': len(train_data.low_occ_species_idx),
                'model': model_setup, #'receptive_fields': receptive_fields,
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
                if random_bg_path is None:
                    inputs = po_inputs[0].to(torch.float32).to(dev)
                else:
                    inputs = torch.cat((po_inputs[0], bg_inputs[0]), 0).to(torch.float32).to(dev)

                # if autoencoder:
                #     input_patches = [transforms.CenterCrop(size=aspp.receptive_field)(inputs) for aspp in model.aspp_branches][::-1]
                #     y_pred, xlist_decoded = model(inputs)
                #     y_pred = torch.sigmoid(y_pred)
                # else:
                y_pred = torch.sigmoid(model(inputs))
                    
            if random_bg_path is None:
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
        auc = roc_auc_score(labels, y_pred)
        auc_low_occ = roc_auc_score(labels[:, train_data.low_occ_species_idx], y_pred[:, train_data.low_occ_species_idx])

        # if autoencoder:
        #     avg_val_classif_loss = np.mean(val_classif_loss_list)
        #     avg_val_reconstr_loss = np.mean(val_reconstr_loss_list, axis=0)
        #     print(f"\tVALIDATION LOSS={avg_val_loss} (classification loss={avg_val_classif_loss}, reconstruction loss={avg_val_reconstr_loss}) \nVALIDATION AUC={auc}")
        # else:
        print(f"\tVALIDATION LOSS={avg_val_loss} \nVALIDATION AUC={auc}")

        df = pd.DataFrame(train_data.species_counts, columns=['n_occ']).reset_index().rename(columns={'index':'species'})
        df['auc'] = [roc_auc_score(labels[:,i], y_pred[:,i]) for i in range(labels.shape[1])]
        df.to_csv(f"{modeldir}{run_name}/last_species_auc.csv", index=False)

        if log_wandb:
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_auc": auc, "val_auc_low_occ": auc_low_occ})
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

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", required=True, help="Run name")
    parser.add_argument("-p", "--path_to_config", help="Path to run config file")

    args = parser.parse_args()
    run_name = args.run_name
    path_to_config = args.path_to_config
    if path_to_config is None:
        path_to_config = f"{modeldir}{run_name}/config.json"
    assert os.path.exists(path_to_config)

    with open(path_to_config, "r") as f:
        config = json.load(f)

    config = {k: v if v != "" else None for k,v in config.items()}

    model_setup = {}
    if config['env_model'] is not None: 
        config['env_model']['covariates'] = [eval(f) for f in config['env_model']['covariates']]
        model_setup['env'] = config['env_model']
    if config['sat_model'] is not None: 
        config['sat_model']['covariates'] = [eval(f) for f in config['sat_model']['covariates']]
        model_setup['sat'] = config['sat_model']

    train_model(
        run_name, 
        log_wandb = config['log_wandb'], 
        model_setup = model_setup,
        wandb_project = config['wandb_project'],
        wandb_id = config['wandb_id'], 
        train_occ_path = eval(config['train_occ_path']), 
        random_bg_path = eval(config['random_bg_path']) if config['random_bg_path'] is not None else None, 
        val_occ_path = eval(config['val_occ_path']), 
        n_max_low_occ = config['n_max_low_occ'],
        embed_shape = config['embed_shape'],
        loss = config['loss'], 
        lambda2 = config['lambda2'],
        n_epochs = config['n_epochs'], 
        batch_size = config['batch_size'], 
        learning_rate = config['learning_rate'], 
        weight_decay = config['weight_decay'],
        num_workers_train = config['num_workers_train'],
        num_workers_val = config['num_workers_val'],
        seed = config['seed'])
    

# run_name = '0510_multimodel_multires'
# path_to_config = f"{modeldir}{run_name}/config.json"
# with open(path_to_config, "r") as f: 
#     config = json.load(f)
# config = {k: v if v != "" else None for k,v in config.items()}
# model_setup = {}

# if config['env_model'] is not None: 
#     config['env_model']['covariates'] = [eval(f) for f in config['env_model']['covariates']]
#     model_setup['env'] = config['env_model']

# if config['sat_model'] is not None: 
#     config['sat_model']['covariates'] = [eval(f) for f in config['sat_model']['covariates']]
#     model_setup['sat'] = config['sat_model']

# train_occ_path = eval(config['train_occ_path'])
# random_bg_path = eval(config['random_bg_path']) if config['random_bg_path'] is not None else None
# val_occ_path = eval(config['val_occ_path'])
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
