from train_model import *

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", required=True, help="Run name")
    parser.add_argument("-p", "--path_to_config", help="Path to run config file")

    args = parser.parse_args()
    run_name = args.run_name
    path_to_config = args.path_to_config

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
    n_max_low_occ = config['n_max_low_occ']
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

    train_data, val_data, model, optimizer, multimodal, autoencoder = setup_model(
        env_model=env_model,
        sat_model=sat_model,
        dataset=dataset,
        random_bg=random_bg,
        n_max_low_occ=n_max_low_occ,
        embed_shape=embed_shape, 
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        seed=seed) 
    model = model.to(dev)

    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=num_workers_val)
    loss_fn = eval(loss)
    species_weights = torch.tensor(train_data.species_weights).to(dev)
    val_loss_fn = torch.nn.BCELoss()

    checkpoint = torch.load(f"{modeldir}{run_name}/best_val_auc.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    max_val_auc = checkpoint['val_auc']
    print(epoch, max_val_auc)

    # evaluate model on validation set
    model.eval()
    val_loss_list, val_classif_loss_list, val_reconstr_loss_list, labels_list, y_pred_list = [], [], [], [], []
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

    np.save(f"{modeldir}{run_name}/y_pred_best_val_auc.pth", y_pred)
    np.save(f"{modeldir}{run_name}/y_true.pth", labels)