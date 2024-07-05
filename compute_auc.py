from train_model import *
from sklearn.metrics import roc_auc_score

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", required=True, help="Run name")
    parser.add_argument("-m", "--model", choices=['best', 'last', 'both'], default='best', help="Use model at best val AUC epoch or last epoch. Options: best, last")

    args = parser.parse_args()
    run_name = args.run_name
    model = args.model
    model_list = ['best_val_auc'] if model == 'best' else (['last'] if model == 'last' else ['best_val_auc','last']) 

    for model_to_load in model_list:
        checkpoint = torch.load(f"{modeldir}{run_name}/{model_to_load}.pth")
        epoch = checkpoint['epoch'] +1
        val_auc = checkpoint['val_auc']
        print(epoch, val_auc)
    
        y_pred =  np.load(f"models/{run_name}/y_pred_{model_to_load}.npy")
        labels = np.load(f"models/{run_name}/y_true.npy")

        y_pred_torch = torch.from_numpy(y_pred.T)
        labels_torch = torch.from_numpy(labels.T)
        auc_list = binary_auroc(y_pred_torch, labels_torch, num_tasks=labels_torch.shape[0])
        print(f"Median AUC (species-wise) {auc_list.median().item()}")
        print(roc_auc_score(labels, y_pred))

        y_pred_torch = torch.from_numpy(y_pred)
        labels_torch = torch.from_numpy(labels)
        auc_list = binary_auroc(y_pred_torch, labels_torch, num_tasks=labels_torch.shape[0])
        print(f"Median AUC (site-wise) {auc_list.median().item()}")