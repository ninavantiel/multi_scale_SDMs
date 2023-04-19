#import random
import torch
from sklearn.metrics import roc_auc_score

from GLC23PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider, JpegPatchProvider
from GLC23Datasets import PatchesDataset, PatchesDatasetMultiLabel
from models import cnn

# SAMPLE DATA
# # data_path = 'data/sample_data/' # root path of the data
# # presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train_sample.csv'
# # presence_absence_path = data_path+'Presence_Absences_occurrences/Presences_Absences_train_sample.csv'

# OCCURRENCE DATA
data_path = 'data/full_data/'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train.csv'
# # presence_absence_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'

# COVARIATES
# p_rgb = JpegPatchProvider(data_path+'SatelliteImages/')#, dataset_stats='jpeg_patches_sample_stats.csv') # take all sentinel imagery layer (4)

# p_soil = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Soilgrids/') #9
# 

# n_features = 15 #44 

if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)

    p_bioclim = MultipleRasterPatchProvider(
        data_path+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
    ) #19
    # p_elevation = RasterPatchProvider(data_path + 'EnvironmentalRasters/Elevation/ASTER_Elevation.tif') #1
    p_hfp_d = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/detailed/') #14
    p_hfp_s = RasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif') #1

    presence_only = PatchesDatasetMultiLabel(
        occurrences=presence_only_path, 
        providers=(p_bioclim, p_hfp_d, p_hfp_s), 
        device=dev
    )
    print(f"\nTRAINING DATA: n={len(presence_only)}")
    print(f"size of input for one sample: {presence_only[0][0].cpu().detach().shape}")
    print(f"size of output (labels) for one sample: {presence_only[0][1].cpu().detach().shape}")

    n_features = presence_only[0][0].cpu().detach().shape[0]
    print(n_features)

    # # get presence-absence date -> val
    # # presence_absence = PatchesDatasetMultiLabel(occurrences=presence_absence_path, providers=(p_rgb, p_bioclim, p_hfp_d, p_hfp_s), device=dev)
    # # print(f"\nVALIDATION DATA: n={len(presence_absence)}")
    # # print(f"size of input for one sample: {presence_absence[0][0].shape}")
    # # print(f"size of output (labels) for one sample: {presence_absence[0][1].shape}")

    train_loader = torch.utils.data.DataLoader(presence_only, shuffle=True, batch_size=2)
    print(train_loader.batch_size)
    print(len(train_loader.dataset))
    # val_loader = torch.utils.data.DataLoader(presence_absence, shuffle=True, batch_size=len(presence_absence))

    model = cnn(n_features).to(dev)
    print(f"\nMODEL: {model}")

    loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight)).to(device)(pred, target)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    n_epochs = 1

    avg_aucs = []
    for epoch in range(n_epochs):
        print(f"EPOCH {epoch}")
        for inputs, labels in train_loader:
            # forward pass
            y_pred = model(inputs)
            print(labels.shape, y_pred.shape)
            loss = loss_fn(y_pred, labels)
            print("LOSS: ", loss)

            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #     for inputs, labels in val_loader:
    #         y_pred = model(inputs)

    #         y_pred = y_pred.detach().numpy()
    #         labels = labels.detach().numpy()

    #         # labels_non_zero = labels[:, labels.sum(axis=0) != 0]
    #         # y_pred_non_zero = y_pred[:, labels.sum(axis=0) != 0]
    #         # if y_pred_non_zero.shape[1] != y_pred.shape[1]: break
    #         # if y_pred_non_zero.shape[0] != y_pred.shape[0]: break

    #         auc_rocs = roc_auc_score(labels, y_pred, average=None)
    #         avg_aucs.append(auc_rocs.mean())
    #         print(f"{epoch}) AVG_AUC={auc_rocs.mean()}, {avg_aucs}")
