# Multi-Scale and Multimodal Species Distribution Modeling

Code accompanying the paper "Multi-Scale and Multimodal Species Distribution Modeling" by van Tiel et al. published at the CV4E workshop at ECCV 2024. 
Parts of the code were adapted from the GeoLifeCLEF 2023 (GLC23) code (https://github.com/plantnet/GeoLifeCLEF). 
All data used with this codebase can be found through the GLC23 Kaggle page (https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/overview).

Single- or multi-scale models with one or two modalities can be trained and evaluated on local data with the script `train_model.py`. 
The script `predict_on_test_set.py` is used to generate predictions on a hidden test set which can be evaluated through the GLC23 Kaggle page.
Models are defined in `models.py`, loss functions in `losses.py` and additional functions in `util.py`.
The dataset is accessed through classes defined in `data/Datasets.py` and `data/PatchesProviders.py`.

Models are defined in a `config.json` file. 
An example is provided in `example_model/config.json` for a multi-scale bi-modal model (environmental covariates with scales 1x1 and 5x5 pixels, satellite images with scales 25x25, 59x59, and 115x115 pixels).
