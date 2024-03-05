import argparse
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
        path_to_config = f"{modeldir}{run_name}/config.txt"
    print(path_to_config)
    assert os.path.exists(path_to_config)

    with open(path_to_config) as f:
        lines = f.readlines()
        dict = {}
        for line in lines:
            line = line.strip().split('=')
            assert len(line) == 2
            dict[line[0]] = line[1]

    model_setup = {}
    if eval(dict['env_model']) is not None: model_setup['env'] = eval(dict['env_model'])
    if eval(dict['sat_model']) is not None: model_setup['sat'] = eval(dict['sat_model'])
    print(model_setup)

    train_model(
        run_name, 
        log_wandb = bool(eval(dict['log_wandb'])), 
        model_setup = model_setup,
        wandb_project = str(eval(dict['wandb_project'])),
        wandb_id = eval(dict['wandb_id']), 
        train_occ_path = eval(dict['train_occ_path']), 
        random_bg_path = eval(dict['random_bg_path']), 
        val_occ_path = eval(dict['val_occ_path']), 
        n_max_low_occ = int(eval(dict['n_max_low_occ'])),
        embed_shape = eval(dict['embed_shape']),
        loss = eval(dict['loss']), 
        lambda2 = eval(dict['lambda2']),
        n_epochs = int(eval(dict['n_epochs'])), 
        batch_size = int(eval(dict['batch_size'])), 
        learning_rate = float(eval(dict['learning_rate'])), 
        seed = int(eval(dict['seed']))
    )
