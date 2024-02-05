from train_model import *

if __name__ == "__main__": 
    # mlp
    train_model(
        run_name='0205_MLP_env_16x16_weighted_loss_05',
        train_occ_path=po_path, 
        random_bg_path=bg_path,
        val_occ_path=pa_path,
        patch_size = 16,
        covariates=['bioclim', 'soil', 'landcover'],
        loss = 'weighted_loss',
        lambda2=0.5
    )

    # cnn
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