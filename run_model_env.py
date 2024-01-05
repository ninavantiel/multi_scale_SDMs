import torch
from tqdm import tqdm
import numpy as np

from data.PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider
from data.Datasets import PatchesDataset
from models import MLP

# paths to data
datadir = 'data/full_data/'
bioclim_dir = datadir+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
soil_dir = datadir+'EnvironmentalRasters/Soilgrids/'
po_path = datadir+'Presence_only_occurrences/Presences_only_train.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

# device (cpu ou cuda)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {dev}\n")

# hyperparameters
batch_size = 64
learning_rate = 1e-4
n_epochs = 10

if __name__ == "__main__":
    # load patch providers for covariates
    print("Making patch providers for predictor variables...")
    p_bioclim = MultipleRasterPatchProvider(bioclim_dir, size=1)
    p_soil = MultipleRasterPatchProvider(soil_dir, size=1)

    # validation data: presence absence data 
    print("Making dataset for presence-absence validation data...")
    val_data = PatchesDataset(occurrences=pa_path, providers=(p_bioclim, p_soil))
    print(f"\nVALIDATION DATA: n_items={len(val_data)}, n_species={len(val_data.species)}")
    print(val_data[0][0].shape, val_data[0][1].shape)
    
    # train data: presence only data 
    print("Making dataset for presence-only training data...")
    train_data = PatchesDataset(occurrences=po_path, providers=(p_bioclim, p_soil), species=val_data.species)
    print(f"\nTRAINING DATA: n_items={len(train_data)}, n_species={len(train_data.species)}")
    n_features = train_data[0][0].shape[0]
    n_species = len(train_data.species)
    print(train_data[0][0].shape, train_data[0][1].shape)
    print(n_features, n_species)

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=8)

    # model and optimizer
    model = MLP(n_features, n_species, 5, 1000).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss function
    loss_fn = torch.nn.BCEWithLogitsLoss().to(dev) # add weights to loss?

    # model training
    start_epoch = 0
    for epoch in range(start_epoch, n_epochs):
        print(f"EPOCH {epoch}")

        model.train()
        train_loss_list = []
        for inputs, labels in tqdm(train_loader):

            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(torch.float32).to(dev) 

            # forward pass
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            train_loss_list.append(loss.cpu().detach())

            # backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = np.array(train_loss_list).mean()
        print(f"{epoch}) TRAIN LOSS={avg_train_loss}")

        model.eval()
        val_loss_list, val_f1_list = [], []
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(torch.float32).to(dev)
            labels = labels.to(dev) #.to(torch.float32)
            y_pred = model(inputs)

            # validation loss
            val_loss = loss_fn(y_pred, labels).cpu().detach()
            val_loss_list.append(val_loss)

            # y_true = labels.cpu().detach().numpy()
            # y_pred =  y_pred.cpu().detach().numpy()
            # y_bin = np.where(y_pred > 0.5, 1, 0)
            # f1 = f1_score(y_true, y_bin, average='macro', zero_division=0)
            # val_f1_list.append(f1)
    
        avg_val_loss = np.array(val_loss_list).mean()
        # avg_val_f1 = np.array(val_f1_list).mean()
        print(f"\tVALIDATION LOSS={avg_val_loss}")#, F1-SCORE={avg_val_f1} (threshold=0.5)")

        # model checkpoint
        # torch.save({
        #     'epoch': epoch,
        #     'state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'train_loss': avg_train_loss,
        #     'val_loss': avg_val_loss
        # }, f"models/{run_name}/last.pth") 

        # save best models
        # if epoch == 0: 
        #     min_train_loss = avg_train_loss
        #     min_val_loss = avg_val_loss
            
        # if avg_train_loss <= min_train_loss:
        #     min_train_loss = avg_train_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_loss': avg_train_loss,
        #         'val_loss': avg_val_loss,
        #         'val_f1': avg_val_f1
        #     }, f"models/{run_name}/best_train_loss.pth")  

        # if avg_val_loss <= min_val_loss:
        #     min_val_loss = avg_val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_loss': avg_train_loss,
        #         'val_loss': avg_val_loss,
        #         'val_f1': avg_val_f1
        #     }, f"models/{run_name}/best_val_loss.pth")  


    