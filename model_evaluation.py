from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import binarize
# import matplotlib.pyplot as plt
# import seaborn as sns

from train_model import *

def compute_f1(labels, pred):
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    f1 = tp / (tp + ((fp+fn)/2))
    return f1

def eval_model(
    run_name, 
    model_setup,
    checkpoint_to_load='last',
    train_occ_path=po_path,
    random_bg_path=None,
    val_occ_path=pa_path,
    n_max_low_occ=50,
    embed_shape=None,
    learning_rate=1e-3,
    seed=42
): 
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev}")
    
    train_data, val_data, model, optimizer, multires = setup_model(
        model_setup, train_occ_path, random_bg_path, val_occ_path,
        n_max_low_occ, embed_shape, learning_rate, seed
    )
    model = model.to(dev)

    print(f"\nLoading model from checkpoint {run_name}")
    checkpoint = torch.load(f"models/{run_name}/{checkpoint_to_load}.pth")
    print(checkpoint['epoch'], checkpoint['val_auc'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

    # data loader
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=128)

    print('\nEvaluating validation data...')
    model.eval()
    labels_list, y_pred_list = [], []
    for inputs, _, labels in tqdm(val_loader):
        if multires:
            inputsA = inputs[0].to(torch.float32).to(dev)
            inputsB = inputs[1].to(torch.float32).to(dev)
            y_pred = torch.sigmoid(model(inputsA, inputsB))
        else:
            inputs = inputs[0].to(torch.float32).to(dev)
            y_pred = torch.sigmoid(model(inputs))
        
        y_pred_list.append(y_pred.cpu().detach().numpy())
        labels_list.append(labels)

    labels = np.concatenate(labels_list)
    y_pred = np.concatenate(y_pred_list)

    auc = roc_auc_score(labels, y_pred)
    print('AUC = ', auc)
    auc_low_occ = roc_auc_score(labels[:, train_data.low_occ_species_idx], y_pred[:, train_data.low_occ_species_idx])
    print('AUC (low occ) = ', auc_low_occ)

    df = pd.DataFrame(train_data.species_counts, columns=['n_occ']).reset_index().rename(columns={'index':'species'})
    df['auc'] = [roc_auc_score(labels[:,i], y_pred[:,i]) for i in range(labels.shape[1])]
    df.to_csv(f"models/{run_name}/{checkpoint_to_load}_auc.csv", index=False)

    f1_scores = {}
    for thresh in np.arange(0.05, 1, 0.05):
        y_pred_bin = binarize(y_pred, threshold=thresh)
        f1_list = [compute_f1(labels[i,:], y_pred_bin[i,:]) for i in range(labels.shape[0])]
        f1_mean = np.mean(f1_list)
        print(thresh, '.... f1 = ', f1_mean)
        f1_scores[thresh] = f1_mean

    max_f1 = np.max(list(f1_scores.values()))
    threshold = [k for k,v in f1_scores.items() if v == max_f1][0]

    # text file
    f = open(f"models/{run_name}/{checkpoint_to_load}_eval.txt", "a")
    f.write(f"epoch = \t{checkpoint['epoch']}\n")
    f.write(f"val AUC = \t{auc}\n")
    f.write(f"val AUC (low occ) = \t{auc_low_occ}\n")
    f.write(f"max F1 = \t{max_f1} (threshold = {threshold})\n")
    f.close()

if __name__ == "__main__":
    eval_model(
        run_name = '0205_CNN_env_16x16_weighted_loss_05',
        model_setup= {'env': {
            'model_name':'CNN', 
            'covariates':[bioclim_dir, soil_dir, landcover_path],
            'patch_size': 16, 
            'n_conv_layers': 2, 
            'n_filters': [32, 64],
            'width': 1000, 
            'kernel_size': 3,
            'pooling_size': 1, 
            'dropout': 0
        }},
        train_occ_path=po_path,
        random_bg_path=bg_path,
        val_occ_path=pa_path
    )
        
    # eval_model(
    #     run_name = '0206_MLP_env_4x4_weighted_loss_05',
    #     model_setup= {'env': {
    #         'model_name':'MLP', 
    #         'covariates':[bioclim_dir, soil_dir, landcover_path],
    #         'patch_size': 4, 
    #         'n_layers': 5, 
    #         'width': 1000, 
    #         'dropout': 0
    #     }},
    #     train_occ_path=po_path,
    #     random_bg_path=bg_path,
    #     val_occ_path=pa_path
    # )
        
    # eval_model(
    #     run_name = '0201_MLP_env_1x1_weighted_loss_05_all_PA_species_with_pseudoabsences',
    #     model_setup= {'env': {
    #         'model_name':'MLP', 
    #         'covariates':[bioclim_dir, soil_dir, landcover_path],
    #         'patch_size': 1, 
    #         'n_layers': 5, 
    #         'width': 1000, 
    #         'dropout': 0
    #     }},
    #     train_occ_path=po_path,
    #     random_bg_path=bg_path,
    #     val_occ_path=pa_path
    # )

    # plots
    # fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained', figsize=(8,3))

    # mean1 = df.auc.mean()
    # ax1.hist(df.auc)
    # ax1.axvline(mean1, color='orange')
    # ax1.set(xlabel='AUC', ylabel='Counts', title=f"Mean AUC = {mean1:.3f}")

    # sns.boxplot(data=df, x="num_presences_cat", y="auc", ax=ax2)
    # ax2.set(xlabel='Nb occurrences', ylabel='AUC')

    # fig.suptitle(run_name)
    # plt.savefig(f"models/{run_name}/{checkpoint_to_load}_eval.png")


    # fig = plt.figure(layout='constrained', figsize=(12, 8))
    # subfigs = fig.subfigures(2, 1)

    # ax1, ax2 = subfigs[0].subplots(1, 2)
    # ax1.scatter(x=df.n_occ, y=df.auc)
    # ax1.set(xlabel='n_occ', ylabel='AUC', title='Nb occurrences vs AUC')

    # ax2.scatter(x=list(f1_scores.keys()), y=list(f1_scores.values()))
    # ax2.axhline(y=max_f1, color='orange')
    # ax2.axvline(x=threshold, color='orange')
    # ax2.set(xlabel='threshold', ylabel='F1', title=f"Threshold vs F1 score\nmax F1 = {max_f1:.5f}")

    # ax1, ax2, ax3 = subfigs[1].subplots(1, 3)
    # ax1.hist(df.auc)
    # mean1 = df.auc.mean()
    # ax1.axvline(mean1, color='orange')
    # ax1.set(xlabel='AUC', ylabel='Counts', title=f"All species\nmean AUC = {mean1:.3f}")

    # ax2.hist(df[df['n_occ'] <= n_max_low_occ].auc)
    # mean2 = df[df['n_occ'] <= n_max_low_occ].auc.mean()
    # ax2.axvline(mean2, color='orange')
    # ax2.set(xlabel='AUC', ylabel='Counts', title=f"Species with n_occ <= 50\nmean AUC = {mean2:.3f}")

    # ax3.hist(df[df['n_occ'] > n_max_low_occ].auc)
    # mean3 = df[df['n_occ'] > n_max_low_occ].auc.mean()
    # ax3.axvline(mean3, color='orange')
    # ax3.set(xlabel='AUC', ylabel='Counts', title=f"Species with n_occ > 50\nmean AUC = {mean3:.3f}")

    # fig.suptitle(run_name)
