import argparse
import json
from train_model import *

modeldir = 'models/'

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", required=True, help="Run name")
    parser.add_argument("-p", "--path_to_config", help="Path to run config file")

    args = parser.parse_args()
    run_name = args.run_name
    print(run_name)
    path_to_config = args.path_to_config
    if path_to_config is None:
        path_to_config = f"{modeldir}{run_name}/config.json" #txt"
    print(path_to_config)
    assert os.path.exists(path_to_config)

    with open(path_to_config, "r") as f:
        config = json.load(f)
        # lines = f.readlines()
        # dict = {}
        # for line in lines:
        #     line = line.strip().split('=')
        #     assert len(line) == 2
        #     dict[line[0]] = line[1]

    config = {k: v if v != "" else None for k,v in config.items()}

    model_setup = {}
    if config['env_model'] is not None: 
        config['env_model']['covariates'] = [eval(f) for f in config['env_model']['covariates']]
        model_setup['env'] = config['env_model']
    if config['sat_model'] is not None: 
        config['sat_model']['covariates'] = [eval(f) for f in config['sat_model']['covariates']]
        model_setup['sat'] = config['sat_model']
    print(model_setup)

    train_model(
        run_name, 
        log_wandb = config['log_wandb'], 
        model_setup = model_setup,
        wandb_project = config['wandb_project'],
        wandb_id = config['wandb_id'], 
        train_occ_path = eval(config['train_occ_path']), 
        random_bg_path = eval(config['random_bg_path']) if config['random_bg_path'] is not None else None, 
        val_occ_path = eval(config['val_occ_path']), 
        n_max_low_occ = config['n_max_low_occ'],
        embed_shape = config['embed_shape'],
        loss = config['loss'], 
        lambda2 = config['lambda2'],
        n_epochs = config['n_epochs'], 
        batch_size = config['batch_size'], 
        learning_rate = config['learning_rate'], 
        seed = config['seed']
    )
