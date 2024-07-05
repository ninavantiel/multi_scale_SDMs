from train_model import *
from sklearn.metrics import f1_score, roc_auc_score

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", required=True, help="Run name")
    parser.add_argument("-m", "--model", choices=['best', 'last', 'both'], default='both', help="Use model at best val AUC epoch or last epoch. Options: best, last")
    parser.add_argument("-p", "--path_to_config", help="Path to run config file")
    parser.add_argument("-t", "--threshold", default='best', help="Find best threshold on val data or use provided numerical value")

    args = parser.parse_args()
    run_name = args.run_name
    path_to_config = args.path_to_config
    model = args.model
    model_list = ['best_val_auc'] if model == 'best' else (['last'] if model == 'last' else ['best_val_auc','last']) 
    threshold = args.threshold

    # read config file
    if path_to_config is None:
        path_to_config = f"{modeldir}{run_name}/config.json"
    assert os.path.exists(path_to_config)
    with open(path_to_config, "r") as f:
        config = json.load(f)
    config = {k: v if v != "" else None for k,v in config.items()}

    # get variables from config 
    log_wandb = config['log_wandb']
    wandb_project = config['wandb_project']
    wandb_id = config['wandb_id']
    env_model = config['env_model']
    sat_model = config['sat_model']
    dataset = config['dataset']
    random_bg = config['random_bg']
    embed_shape = config['embed_shape']
    loss = config['loss']
    lambda2 = config['lambda2']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_workers_train = config['num_workers_train']
    num_workers_val = config['num_workers_val']
    seed = config['seed']

    seed_everything(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev}")

    train_data, val_data, test_data, model, optimizer, multimodal, autoencoder = setup_model(
        env_model=env_model,
        sat_model=sat_model,
        dataset=dataset,
        random_bg=random_bg,
        embed_shape=embed_shape, 
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        seed=seed,
        test=True) 
    model = model.to(dev)

    for model_to_load in model_list:

        checkpoint = torch.load(f"{modeldir}{run_name}/{model_to_load}.pth")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] +1
        val_auc = checkpoint['val_auc']
        print(epoch, val_auc)

        if os.path.exists(f"models/{run_name}/y_pred_epoch_{str(epoch)}.npy") and os.path.exists(f"models/{run_name}/y_true.npy"):
            print("Loading y_pred and y_true...")
            y_pred =  np.load(f"models/{run_name}/y_pred_epoch_{str(epoch)}.npy")
            labels = np.load(f"models/{run_name}/y_true.npy")
        
        else:
            val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=num_workers_val)
            # evaluate model on validation set
            model.eval()
            labels_list, y_pred_list = [], []
            for inputs, _, labels in tqdm(val_loader):
                labels = labels.to(torch.float32).to(dev) 
                labels_list.append(labels.cpu().detach().numpy())

                if multimodal:
                    inputsA = inputs[0].to(torch.float32).to(dev)
                    inputsB = inputs[1].to(torch.float32).to(dev)
                    y_pred = torch.sigmoid(model(inputsA, inputsB))
                else:
                    inputs = inputs[0].to(torch.float32).to(dev)
                    y_pred = torch.sigmoid(model(inputs))

                y_pred_list.append(y_pred.cpu().detach().numpy())

            labels = np.concatenate(labels_list)
            y_pred = np.concatenate(y_pred_list)

            np.save(f"{modeldir}{run_name}/y_pred_epoch_{str(epoch)}.npy", y_pred)
            np.save(f"{modeldir}{run_name}/y_true.npy", labels)

        df = pd.DataFrame(train_data.species_counts, columns=['n_occ']).reset_index().rename(columns={'index':'species'})
        df['auc'] = [roc_auc_score(labels[:,i], y_pred[:,i]) for i in range(labels.shape[1])]
        df.to_csv(f"{modeldir}{run_name}/species_auc_epoch_{str(epoch)}.csv", index=False)
        print(f"Median AUC across species = {df['auc'].median()}")
        
        site_auc = [roc_auc_score(labels[i,:], y_pred[i,:]) for i in range(labels.shape[0])]
        print(f"Median AUC across sites = {np.median(site_auc)}")

        if threshold == 'best':
            f1_scores = []
            thresholds = np.arange(0.2, 0.8, 0.025)
            for thresh in tqdm(thresholds):
                y_bin = np.where(y_pred > thresh, 1, 0)
                f1_scores.append(f1_score(labels.T, y_bin.T, average='macro', zero_division=0))

            val_threshold = thresholds[np.argmax(f1_scores)]
            val_f1 = np.max(f1_scores)      
            print(f"Best threshold={val_threshold} --> validation F1-score={val_f1}")
            np.save(f"{modeldir}{run_name}/f1_scores_{model_to_load}.npy", np.stack([thresholds, f1_scores]))

        else:
            val_threshold = float(threshold)
            assert val_threshold >= 0 and val_threshold <= 1
            y_bin = np.where(y_pred > val_threshold, 1, 0)
            val_f1 = f1_score(labels.T, y_bin.T, average='macro', zero_division=0)
            print(f"Threshold={val_threshold} --> validation F1-score={val_f1}")

        test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)
        y_pred_list = []
        for inputs, _, _ in tqdm(test_loader):
            if multimodal:
                inputsA = inputs[0].to(torch.float32).to(dev)
                inputsB = inputs[1].to(torch.float32).to(dev)
                y_pred = torch.sigmoid(model(inputsA, inputsB))
            else:
                inputs = inputs[0].to(torch.float32).to(dev)
                y_pred = torch.sigmoid(model(inputs))
            y_pred_list.append(y_pred.cpu().detach().numpy())

        y_pred = np.concatenate(y_pred_list)
        # y_bin = np.where(y_pred > best_threshold, 1, 0)
        targets = train_data.species_pred
        pred_species = [' '.join([str(x) for x in targets[np.where(y_pred[i, :] > val_threshold)]]) for i in range(y_pred.shape[0])]
        sub_df = pd.DataFrame({'Id': test_data.submission_id, 'Predicted': pred_species})
        sub_df.to_csv(f"{modeldir}{run_name}/submission_epoch_{str(epoch)}_thresh_{str(val_threshold)}.csv", index=False)

