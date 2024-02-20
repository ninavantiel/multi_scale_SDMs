from train_model import *

if __name__ == "__main__": 
    train_model(
        run_name='0219_MultiScale_env_1_sat_128x128', 
        log_wandb=True, 
        wandb_project='spatial_extent_glc23_encoder',
        train_occ_path=po_path, 
        val_occ_path=pa_path, 
        model_setup={'env': {
            'model_name':'MLP', 'covariates':[bioclim_dir, soil_dir, landcover_path],
            'patch_size': 1, 'n_layers': 5, 'width': 1280, 'dropout': 0.5
        }, 'sat': {
            'model_name':'ResNet', 'covariates':[sat_dir],
            'patch_size': 128, 'pretrained': True
        }},
        embed_shape=512,
        loss='weighted_loss', 
        lambda2=1,
        n_epochs=100
    )

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