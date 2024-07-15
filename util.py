import os
import random
import numpy as np
import torch

from models import MLP, ShallowCNN, get_resnet, MultimodalModel, MultiResolutionModel, MultiResolutionAutoencoder
from data.PatchesProviders import RasterPatchProvider, MultipleRasterPatchProvider, JpegPatchProvider

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # Numpy seed also uses by Scikit Learn
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_path_to(path_to, datadir):
    dataset_dir = datadir+'GLC23/' 
    paths = {
        "po": 'Presence_only_occurrences/Presences_only_train_sampled_species_in_val.csv',
        "random_bg": 'Presence_only_occurrences/Pseudoabsence_locations_bioclim_soil.csv',
        "pa": 'Presence_Absence_surveys/Presences_Absences_train.csv',
        "test": 'For_submission/test_blind.csv',
        "sat": 'SatelliteImages/',
        "bioclim": 'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/',
        "soil": 'EnvironmentalRasters/Soilgrids/',
        "landcover": 'EnvironmentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif'
    }
    
    path = dataset_dir + paths[path_to]
    return path

def make_providers(covariate_paths_list, patch_size, flatten, id_col='patchID', select=None):
    print(f"\nMaking patch providers with size={patch_size}x{patch_size}, flatten={flatten} for covariates:")
    providers = []
    for cov in covariate_paths_list:
        print(f"\t - {cov}")
        if 'Satellite' in cov:
            if flatten and patch_size != 1: 
                print("jpeg patch provider for satellite images cannot flatten image patches")
                return
            providers.append(JpegPatchProvider(cov, select=select, size=patch_size, id_col=id_col))
        elif '.tif' in cov:
            providers.append(RasterPatchProvider(cov, size=patch_size, flatten=flatten))
        else:
            providers.append(MultipleRasterPatchProvider(cov, size=patch_size, flatten=flatten))
    return providers

def make_model(model_dict):
    param_names = {'input_shape', 'output_shape', 'backbone', 'patch_size', 'backbone_params', 'spatial_block_params'}
    assert param_names.issubset(set(model_dict.keys()))

    assert model_dict['backbone'] in ['CNN', 'ResNet']
    if model_dict['backbone'] == 'CNN':
        backbone_param_names = {'n_filters', 'kernel_sizes', 'paddings', 'pooling_sizes'}
    elif model_dict['backbone'] == 'ResNet':
        backbone_param_names = {'pretrained'}
    assert backbone_param_names.issubset(set(model_dict['backbone_params'].keys()))

    spatial_block_param_names = {'out_channels', 'out_size', 'kernel_sizes', 'strides', 'pooling_sizes', 'n_linear_layers'}
    assert spatial_block_param_names.issubset(set(model_dict['spatial_block_params'].keys()))

    if model_dict['model_name'] == 'MultiResolutionModel':
        model = MultiResolutionModel(
            model_dict['input_shape'][0],
            model_dict['patch_size'],
            model_dict['output_shape'],
            model_dict['backbone'],
            model_dict['backbone_params'], 
            model_dict['spatial_block_params'])

    return model
