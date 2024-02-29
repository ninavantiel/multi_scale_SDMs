import argparse
from train_model import *

modeldir = 'models/'

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", required=True, help="Run name")
    parser.add_argument("-p", "--path_to_config", help="Path to run config file")

    args = parser.parse_args()
    run_name = args.run_name
    print(run_name)
    path_to_config = args.path_to_config
    if path_to_config is None:
        path_to_config = f"{modeldir}/{run_name}/config.txt"
    print(path_to_config)
    assert os.path.exists(path_to_config)

    with open(path_to_config) as f:
        lines = f.readlines()
        dict = {}
        for line in lines:
            line = line.strip().split('=')
            assert len(line) == 2
            dict[line[0]] = line[1]

    model_setup = {}
    if dict['env_model'] is not None: model_setup['env'] = eval(dict['env_model'])
    if dict['sat_model'] is not None: model_setup['sat'] = eval(dict['sat_model'])

    train_model(
        run_name, 
        log_wandb = bool(eval(dict['log_wandb'])), 
        model_setup = model_setup,
        wandb_project = str(eval(dict['wandb_project'])),
        wandb_id = eval(dict['wandb_id']), 
        train_occ_path = eval(dict['train_occ_path']), 
        random_bg_path = eval(dict['random_bg_path']), 
        val_occ_path = eval(dict['val_occ_path']), 
        n_max_low_occ = int(eval(dict['n_max_low_occ'])),
        embed_shape = eval(dict['embed_shape']),
        loss = eval(dict['loss']), 
        lambda2 = eval(dict['lambda2']),
        n_epochs = int(eval(dict['n_epochs'])), 
        batch_size = int(eval(dict['batch_size'])), 
        learning_rate = float(eval(dict['learning_rate'])), 
        seed = int(eval(dict['seed']))
    )


#   train_model(
#         run_name='0221_resnet_sat_128x128_an_full_loss', 
#         log_wandb=True, 
#         wandb_project='spatial_extent_glc23_encoder',
#         train_occ_path=po_path, 
#         val_occ_path=pa_path, 
#         model_setup={'sat': {
#             'model_name':'ResNet', 'covariates':[sat_dir],
#             'patch_size': 128, 'pretrained': True
#         }},
#         n_epochs=100, 
#         loss='an_full_loss'
#     )

#   train_model(
#         run_name='0221_resnet_sat_128x128_not_pretrained', 
#         log_wandb=True, 
#         wandb_project='spatial_extent_glc23_sample50',
#         train_occ_path=po_path_sampled_50, 
#         val_occ_path=pa_path, 
#         model_setup={'sat': {
#             'model_name':'ResNet', 'covariates':[sat_dir],
#             'patch_size': 128, 'pretrained': False
#         }},
#         n_epochs=100
#     )

    # train_model(
    #     run_name='0219_MultiScale_env_1_sat_128x128', 
    #     log_wandb=True, 
    #     wandb_project='spatial_extent_glc23_encoder',
    #     train_occ_path=po_path, 
    #     val_occ_path=pa_path, 
    #     model_setup={'env': {
    #         'model_name':'MLP', 'covariates':[bioclim_dir, soil_dir, landcover_path],
    #         'patch_size': 1, 'n_layers': 5, 'width': 1280, 'dropout': 0.5
    #     }, 'sat': {
    #         'model_name':'ResNet', 'covariates':[sat_dir],
    #         'patch_size': 128, 'pretrained': True
    #     }},
    #     embed_shape=512,
    #     loss='weighted_loss', 
    #     lambda2=1,
    #     n_epochs=100
    # )

    # train_model(
    #     run_name='0216_ResNet_env_128x128', 
    #     log_wandb=True, 
    #     wandb_project='spatial_extent_glc23_encoder',
    #     train_occ_path=po_path, 
    #     val_occ_path=pa_path, 
    #     model_setup={'env': {
    #         'model_name':'ResNet', 'covariates':[bioclim_dir, soil_dir, landcover_path],
    #         'patch_size': 128, 'pretrained': False
    #     }},
    #     loss='weighted_loss', 
    #     lambda2=1,
    #     n_epochs=100
    # )

    # train_model(
    #     run_name='0216_resnet_sat_32x32_pretrained', 
    #     log_wandb=True, 
    #     wandb_project='spatial_extent_glc23_sample50',
    #     train_occ_path=po_path_sampled_50, 
    #     val_occ_path=pa_path, 
    #     model_setup={'sat': {
    #         'model_name':'ResNet', 'covariates':[sat_dir],
    #         'patch_size': 32, 'pretrained': True
    #     }},
    #     n_epochs=100
    # )

    # train_model(
    #     run_name='0215_CNN_env_128x128', 
    #     log_wandb=True, 
    #     wandb_project='spatial_extent_glc23_encoder',
    #     train_occ_path=po_path, 
    #     val_occ_path=pa_path, 
    #     model_setup={'env': {
    #         'model_name':'CNN', 'covariates':[bioclim_dir, soil_dir, landcover_path],
    #         'patch_size': 64, 'n_conv_layers': 3, 'n_filters': [32, 64, 64],
    #         'width': 1280, 'kernel_size': 3, 'pooling_size': 2, 'dropout': 0.5
    #     }},
    #     loss='weighted_loss', 
    #     lambda2=1,
    #     n_epochs=100
    # )

    # train_model(
    #     run_name='0215_resnet_sat_224x224_pretrained', 
    #     log_wandb=True, 
    #     wandb_project='spatial_extent_glc23_sample50',
    #     train_occ_path=po_path_sampled_50, 
    #     val_occ_path=pa_path, 
    #     model_setup={'sat': {
    #         'model_name':'ResNet', 'covariates':[sat_dir],
    #         'patch_size': 224
    #     }},
    #     n_epochs=100
    # )

    # train_model(
    #     run_name='0214_MLP_env_1x1', 
    #     log_wandb=True, 
    #     wandb_project='spatial_extent_glc23_sample50',
    #     train_occ_path=po_path_sampled_50, 
    #     val_occ_path=pa_path, 
    #     model_setup={'env': {
    #         'model_name':'MLP', 'covariates':[bioclim_dir, soil_dir, landcover_path],
    #         'patch_size': 1, 'n_layers': 5, 'width': 1280, 'dropout': 0.5
    #     }},
    #     n_epochs=100
    # )

    # train_model(
    #     run_name='0214_resnet_sat_128x128_pretrained', 
    #     log_wandb=True, 
    #     wandb_project='spatial_extent_glc23_sample50',
    #     train_occ_path=po_path_sampled_50, 
    #     val_occ_path=pa_path, 
    #     model_setup={'sat': {
    #         'model_name':'ResNet', 'covariates':[sat_dir],
    #         'patch_size': 128
    #     }},
    #     n_epochs=100
    # )

    # train_model(
    #     run_name='0213_resnet_sat_128x128', 
    #     log_wandb=True, wandb_id='8ft04lyr',
    #     wandb_project='spatial_extent_glc23_encoder',
    #     train_occ_path=po_path, 
    #     val_occ_path=pa_path, 
    #     model_setup={'sat': {
    #         'model_name':'ResNet', 'covariates':[sat_dir],
    #         'patch_size': 128, 'pretrained': True
    #     }},
    #     loss='weighted_loss', 
    #     lambda2=1,
    #     n_epochs=100
    # )
    
    # train_model(
    #     run_name='0213_MLP_env_1x1', 
    #     log_wandb=True, 
    #     wandb_project='spatial_extent_glc23_encoder',
    #     train_occ_path=po_path, 
    #     val_occ_path=pa_path, 
    #     model_setup={'env': {
    #         'model_name':'MLP', 'covariates':[bioclim_dir, soil_dir, landcover_path],
    #         'patch_size': 1, 'n_layers': 5, 'width': 1280, 'dropout': 0.5
    #     }},
    #     loss='weighted_loss', 
    #     lambda2=1,
    #     n_epochs=100
    # )

    # train_multiscale_model(
    #     run_name='0208_multiscale_test',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     train_occ_path=po_path_sampled_25,
    #     val_occ_path=pa_path,
    #     patch_sizes=[1, 10],
    #     covariates=[[bioclim_dir], [sat_dir]],
    #     batch_size=64
    # )
  
    # train_model(
    #     run_name='0209_MLP_env_16_weighted_loss_1_bs_128_lr_1e-3',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     train_occ_path=po_path_sampled_25, 
    #     val_occ_path=pa_path,
    #     patch_size = 16, 
    #     covariates=[bioclim_dir, soil_dir, landcover_path],
    #     model='MLP', 
    #     model_params={'n_layers':5, 'width':1280, 'dropout':0.5},
    #     loss = 'weighted_loss', lambda2=1,
    #     batch_size=128, learning_rate=1e-3
    # )

#   train_model(
#         run_name='0209_CNN_env_16_weighted_loss_1_bs_128_lr_1e-3',
#         log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
#         train_occ_path=po_path_sampled_25, 
#         val_occ_path=pa_path,
#         patch_size = 16, 
#         covariates=[bioclim_dir, soil_dir, landcover_path],
#         model='CNN', n_conv_layers=2, n_filters=[32, 64],
#         width=1280, pooling_size=2, dropout=0.5,
#         loss = 'weighted_loss', lambda2=1,
#         batch_size=128, learning_rate=1e-3
#     )

    # train_model(
    #     run_name='0209_MLP_env_4_weighted_loss_1_bs_128_le_1e-3',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     wandb_id='j7s7zutz',
    #     train_occ_path=po_path_sampled_25, 
    #     val_occ_path=pa_path,
    #     patch_size = 4,
    #     covariates=[bioclim_dir, soil_dir, landcover_path],
    #     loss = 'weighted_loss', lambda2=1,
    #     batch_size=128, learning_rate=1e-3
    # )
    
    # train_model(
    #     run_name='0208_CNN_env_32_weighted_loss_1_bs_128_lr_1e-3',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     train_occ_path=po_path_sampled_25, 
    #     val_occ_path=pa_path,
    #     patch_size = 32, 
    #     covariates=[bioclim_dir, soil_dir, landcover_path],
    #     model='CNN', n_conv_layers=2, n_filters=[32, 64],
    #     width=1280, pooling_size=2, dropout=0.5,
    #     loss = 'weighted_loss', lambda2=1,
    #     batch_size=128, learning_rate=1e-3
    # )

    # train_model(
    #     run_name='0208_MLP_env_1_weighted_loss_1_bs_128_lr_1e-3',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     wandb_id='wzu2rzl9',
    #     train_occ_path=po_path_sampled_25, 
    #     val_occ_path=pa_path,
    #     patch_size = 1,
    #     covariates=[bioclim_dir, soil_dir, landcover_path],
    #     model='MLP', n_layers=5, width=1280, dropout=0.5,
    #     loss = 'weighted_loss', lambda2=1,
    #     batch_size=128, learning_rate=1e-3
    # )

    # train_model(
    #     run_name='0208_resnet_sat_128',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     train_occ_path=po_path_sampled_25,
    #     val_occ_path=pa_path,
    #     patch_size = 128, covariates=[sat_dir],
    #     model='ResNet',
    #     loss = 'weighted_loss'
    # )

    # train_model(
    #     run_name='0208_MLP_env_1_weighted_loss_1',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     train_occ_path=po_path_sampled_25, 
    #     val_occ_path=pa_path,
    #     patch_size = 1,
    #     covariates=[bioclim_dir, soil_dir, landcover_path],
    #     model='MLP', n_layers=5, width=1280, dropout=0.5,
    #     loss = 'weighted_loss', lambda2=1,
    # )

    # train_model(
    #     run_name='0207_CNN_env_8x8_weighted_loss_1',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     train_occ_path=po_path_sampled_25, 
    #     val_occ_path=pa_path,
    #     patch_size = 8,
    #     covariates=[bioclim_dir, soil_dir, landcover_path],
    #     model='CNN',
    #     n_conv_layers=2,
    #     n_filters=[32, 64],
    #     width=1280,
    #     pooling_size=1,
    #     dropout=0.5,
    #     loss = 'weighted_loss', lambda2=1,
    # )

    # train_model(
    #     run_name='0207_CNN_env_32x32_weighted_loss_1',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     train_occ_path=po_path_sampled_25, 
    #     val_occ_path=pa_path,
    #     patch_size = 32,
    #     covariates=[bioclim_dir, soil_dir, landcover_path],
    #     model='CNN',
    #     n_conv_layers=2,
    #     n_filters=[32, 64],
    #     width=1280,
    #     pooling_size=2,
    #     dropout=0.5,
    #     loss = 'weighted_loss', lambda2=1,
    # )

    # train_model(
    #     run_name='0206_MLP_env_4x4_weighted_loss_05',
    #     train_occ_path=po_path, 
    #     random_bg_path=bg_path,
    #     val_occ_path=pa_path,
    #     patch_size = 4,
    #     covariates=[bioclim_dir, soil_dir, landcover_path],#['bioclim', 'soil', 'landcover'],
    #     loss = 'weighted_loss',
    #     lambda2=0.5,
    #     batch_size=1024
    # )

    # train_model(
    #     run_name='0205_MLP_env_16x16_weighted_loss_05',
    #     train_occ_path=po_path, 
    #     random_bg_path=bg_path,
    #     val_occ_path=pa_path,
    #     patch_size = 16,
    #     covariates=[bioclim_dir, soil_dir],#, landcover_path],#['bioclim', 'soil', 'landcover'],
    #     loss = 'weighted_loss',
    #     lambda2=0.5
    # )

    # train_model(
    #     run_name='0205_CNN_env_16x16_weighted_loss_05',
    #     train_occ_path=po_path, 
    #     random_bg_path=bg_path,
    #     val_occ_path=pa_path,
    #     patch_size = 16,
    #     covariates=['bioclim', 'soil', 'landcover'],
    #     model='CNN',
    #     n_conv_layers=2,
    #     n_filters=[32, 64],
    #     loss = 'weighted_loss',
    #     lambda2=0.5
    # )