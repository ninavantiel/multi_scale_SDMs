from train_model import *
from train_multiscale_model import *

if __name__ == "__main__": 
    # train_multiscale_model(
    #     run_name='0208_multiscale_test',
    #     log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
    #     train_occ_path=po_path_sampled_25,
    #     val_occ_path=pa_path,
    #     patch_sizes=[1, 10],
    #     covariates=[[bioclim_dir], [sat_dir]],
    #     batch_size=64
    # )
  
    train_model(
        run_name='0209_MLP_env_16_weighted_loss_1_bs_128_lr_1e-3',
        log_wandb=True, wandb_project='spatial_extent_glc23_sample_25',
        train_occ_path=po_path_sampled_25, 
        val_occ_path=pa_path,
        patch_size = 16, 
        covariates=[bioclim_dir, soil_dir, landcover_path],
        model='MLP', 
        model_params={'n_layers':5, 'width':1280, 'dropout':0.5},
        loss = 'weighted_loss', lambda2=1,
        batch_size=128, learning_rate=1e-3
    )

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