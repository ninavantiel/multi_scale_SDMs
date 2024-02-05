import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import binarize


from models import MLP
from losses import an_full_loss, full_weighted_loss
from data.Datasets import PatchesDatasetCooccurrences
from data.PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider

run_name = '0131_MLP_env_1x1_weighted_loss_all_PA_species/'
checkpoint_to_load = 'best_val_loss'#'last' # '
patch_size = 1
flatten = True
learning_rate = 1e-3
n_layers = 5
width = 1000
n_max_low_occ = 50
batch_size = 256
learning_rate = 1e-3

datadir = 'data/full_data/'
po_path = datadir+'Presence_only_occurrences/Presences_only_train_sampled_100_percent_min_1_occurrences.csv' 
bg_path = datadir+'Presence_only_occurrences/Pseudoabsence_locations_bioclim_soil.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

bioclim_dir = datadir+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
soil_dir = datadir+'EnvironmentalRasters/Soilgrids/'
landcover_dir = datadir+'EnvironmentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif'

def compute_f1(labels, pred):
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    f1 = tp / (tp + ((fp+fn)/2))
    return f1

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev}\n")

    print('\nMaking patch providers...')
    p_bioclim = MultipleRasterPatchProvider(bioclim_dir, size=patch_size, flatten=flatten) 
    p_soil = MultipleRasterPatchProvider(soil_dir, size=patch_size, flatten=flatten) 
    p_landcover = RasterPatchProvider(landcover_dir, size=patch_size, flatten=flatten)

    print('\nMaking training dataset...')
    train_data = PatchesDatasetCooccurrences(occurrences=po_path, providers=(p_bioclim, p_soil, p_landcover), pseudoabsences=bg_path)
    print(f"\nTRAINING DATA: n_items={len(train_data)}, n_species={len(train_data.species)}")

    n_features = train_data[0][0].shape[0]
    n_species = len(train_data.species)
    print(f"nb of features = {n_features}")

    low_occ_species = train_data.species_counts[train_data.species_counts <= n_max_low_occ].index
    low_occ_species_idx = np.where(np.isin(train_data.species, low_occ_species))[0]
    print(f"nb of species with less than {n_max_low_occ} occurrences = {len(low_occ_species_idx)}")

    print('\nMaking validation dataset...')
    val_data = PatchesDatasetCooccurrences(occurrences=pa_path, providers=(p_bioclim, p_soil, p_landcover), species=train_data.species)
    print(f"\nVALIDATION DATA: n_items={len(val_data)}, n_species={len(val_data.species)}")
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=4)

    # model and optimizer
    model = MLP(n_features, n_species, n_layers, width).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss function
    # loss_fn = an_full_loss 
    # species_weights = torch.tensor(train_data.species_weights).to(dev)
    # val_loss_fn = torch.nn.BCEWithLogitsLoss()

    print('\nLoading model checkpoint...')
    checkpoint = torch.load(f"models/{run_name}/{checkpoint_to_load}.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print('\nEvaluating validation data...')
    model.eval()
    labels_list, y_pred_list = [], []
    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(torch.float32).to(dev)
        labels = labels.to(torch.float32).to(dev)
        labels_list.append(labels.cpu().detach().numpy())
        y_pred = model(inputs)
        y_pred_sigmoid = torch.sigmoid(y_pred)
        y_pred_list.append(y_pred_sigmoid.cpu().detach().numpy())

    labels = np.concatenate(labels_list)
    y_pred = np.concatenate(y_pred_list)

    auc = roc_auc_score(labels, y_pred)
    print('AUC = ', auc)
    auc_low_occ = roc_auc_score(labels[:, low_occ_species_idx], y_pred[:, low_occ_species_idx])
    print('AUC (low occ) = ', auc_low_occ)

    df = pd.DataFrame(train_data.species_counts, columns=['n_occ']).reset_index()
    df['auc'] = [roc_auc_score(labels[:,i], y_pred[:,i]) for i in range(labels.shape[1])]

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

    # save plots
    fig = plt.figure(layout='constrained', figsize=(12, 8))
    subfigs = fig.subfigures(2, 1)

    ax1, ax2 = subfigs[0].subplots(1, 2)
    ax1.scatter(x=df.n_occ, y=df.auc)
    ax1.set(xlabel='n_occ', ylabel='AUC', title='Nb occurrences vs AUC')

    ax2.scatter(x=list(f1_scores.keys()), y=list(f1_scores.values()))
    ax2.axhline(y=max_f1, color='orange')
    ax2.axvline(x=threshold, color='orange')
    ax2.set(xlabel='threshold', ylabel='F1', title=f"Threshold vs F1 score\nmax F1 = {max_f1:.5f}")

    ax1, ax2, ax3 = subfigs[1].subplots(1, 3)
    ax1.hist(df.auc)
    mean1 = df.auc.mean()
    ax1.axvline(mean1, color='orange')
    ax1.set(xlabel='AUC', ylabel='Counts', title=f"All species\nmean AUC = {mean1:.3f}")

    ax2.hist(df[df['n_occ'] <= n_max_low_occ].auc)
    mean2 = df[df['n_occ'] <= n_max_low_occ].auc.mean()
    ax2.axvline(mean2, color='orange')
    ax2.set(xlabel='AUC', ylabel='Counts', title=f"Species with n_occ <= 50\nmean AUC = {mean2:.3f}")

    ax3.hist(df[df['n_occ'] > n_max_low_occ].auc)
    mean3 = df[df['n_occ'] > n_max_low_occ].auc.mean()
    ax3.axvline(mean3, color='orange')
    ax3.set(xlabel='AUC', ylabel='Counts', title=f"Species with n_occ > 50\nmean AUC = {mean3:.3f}")

    fig.suptitle(run_name)
    plt.savefig(f"models/{run_name}/{checkpoint_to_load}_eval.png")