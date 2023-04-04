#import random
import torch
from sklearn.metrics import roc_auc_score

from data.GLC23PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider, JpegPatchProvider
from data.GLC23Datasets import PatchesDataset, PatchesDatasetMultiLabel
from models import cnn

# data_path = 'data/sample_data/' # root path of the data
data_path = 'data/'

# OCCURRENCE DATA
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train.csv' #_sample.csv'
presence_absence_path = data_path+'Presence_Absences_surveys/Presences_Absences_train.csv'#occurrences/Presences_Absences_train_sample.csv'

# COVARIATES
# configure providers
p_rgb = JpegPatchProvider(data_path+'SatelliteImages/', dataset_stats='jpeg_patches_sample_stats.csv') # take all sentinel imagery layer (4: r,g,b,nir)
p_bioclim = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/') # take all bioclimatic rasters (3: bio1, 2, 11)
#p_soil = MultipleRasterPatchProvider(data_path+'')
#p_elevation =
p_hfp_d = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/detailed/') # take all rasters from human footprint detailed (2: built2009, lights2009)
p_hfp_s = RasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif') # take the human footprint 2009 summurized raster
provider_list = [p_rgb, p_bioclim, p_hfp_d, p_hfp_s]

if __name__ == "__main__":
    # get presence-only data -> train
    presence_only = PatchesDatasetMultiLabel(occurrences=presence_only_path, providers=provider_list)
    print(f"\nTRAINING DATA: n={len(presence_only)}")
    print(f"size of input for one sample: {presence_only[0][0].shape}")
    print(f"size of output (labels) for one sample: {presence_only[0][1].shape}")

    # get presence-absence date -> val
    presence_absence = PatchesDatasetMultiLabel(occurrences=presence_absence_path, providers=provider_list)
    print(f"\nVALIDATION DATA: n={len(presence_absence)}")
    print(f"size of input for one sample: {presence_absence[0][0].shape}")
    print(f"size of output (labels) for one sample: {presence_absence[0][1].shape}")

    train_loader = torch.utils.data.DataLoader(presence_only, shuffle=True, batch_size=64)
    val_loader = torch.utils.data.DataLoader(presence_absence, shuffle=True, batch_size=len(presence_absence))

    model = cnn()
    print(f"\nMODEL: {model}")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    n_epochs = 10

    avg_aucs = []
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            # forward pass
            y_pred = model(inputs)
            print(labels.shape, y_pred.shape)
            loss = loss_fn(y_pred, labels)
            print(loss)

            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for inputs, labels in val_loader:
            y_pred = model(inputs)

            y_pred = y_pred.detach().numpy()
            labels = labels.detach().numpy()

            # labels_non_zero = labels[:, labels.sum(axis=0) != 0]
            # y_pred_non_zero = y_pred[:, labels.sum(axis=0) != 0]
            # if y_pred_non_zero.shape[1] != y_pred.shape[1]: break
            # if y_pred_non_zero.shape[0] != y_pred.shape[0]: break

            auc_rocs = roc_auc_score(labels, y_pred, average=None)
            avg_aucs.append(auc_rocs.mean())
            print(f"{epoch}) AVG_AUC={auc_rocs.mean()}, {avg_aucs}")
