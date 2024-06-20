from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import time

from data.PatchesProviders import MetaPatchProvider

class PatchesDataset(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        species=None,
        label_name='speciesId',
        item_columns=['lat','lon','patchID','dayOfYear'],
    ):
        self.occurrences = Path(occurrences)
        self.label_name = label_name
        self.item_columns = item_columns

        df = pd.read_csv(self.occurrences, sep=";", header='infer', low_memory=False)
        if species is None: 
            self.species = np.unique(df[label_name].values)
        else: 
            self.species = species
            df = df[df['speciesId'].isin(species)]
        self.items = df[item_columns + [label_name]]

        self.base_providers = providers
        self.provider = MetaPatchProvider(self.base_providers)

    def __len__(self):
        return self.items.shape[0]

    def __getitem__(self, index):
        item = self.items.iloc[index][self.item_columns].to_dict()

        item_species = self.items.query(
            ' and '.join([f'{k} == {v}' for k, v in item.items()])
        )[self.label_name].values
        labels = 1 * np.isin(self.species, item_species)

        patch = self.provider[item]

        return patch, labels

class PatchesDatasetCooccurrences(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        species=None,
        label_name='speciesId',
        item_columns=['lat','lon','patchID', 'dayOfYear'],
        pseudoabsences=None,
        n_low_occ=50,
        sep=';',
        test=False
    ):
        self.occurrences = Path(occurrences)
        self.label_name = label_name
        self.item_columns = item_columns
        self.pseudoabsences = pseudoabsences
        self.n_low_occ = n_low_occ
        self.test = test

        df = pd.read_csv(self.occurrences, sep=sep, header='infer', low_memory=False)
        
        if self.test:
            df = df.reset_index()
            label_name = 'index'

        self.items = pd.DataFrame(df.groupby(item_columns)[label_name].agg(list)).reset_index()
        self.n_items = self.items.shape[0]
        print(f"nb items = {self.n_items}")

        if self.pseudoabsences is not None:
            self.pseudoabsence_items = pd.read_csv(self.pseudoabsences).sample(self.n_items)
            print(f"nb pseudoabsences = {self.pseudoabsence_items.shape[0]}")
        
        self.base_providers = providers
        self.providers = [MetaPatchProvider(p) for p in self.base_providers]

        if self.test:
            self.submission_id = df.Id

        else:
            self.species_data = np.unique(df[label_name].values)
            self.n_species_data = len(self.species_data)

            if species is None: 
                self.species_pred = self.species_data
                self.species_counts = pd.Series([sps for sps_list in self.items[label_name] for sps in sps_list]).value_counts().sort_index()
                self.species_weights = (self.n_items / self.species_counts).values

            else: 
                self.species_pred = species
                self.species_pred_in_data = [s in self.species_data for s in self.species_pred]
                self.n_species_pred_in_data = np.sum(self.species_pred_in_data)
            
            self.n_species_pred = len(self.species_pred)

    def __len__(self):
        return self.items.shape[0]

    def __getitem__(self, index):
        item = self.items.iloc[index][self.item_columns].to_dict()
        patches = [p[item] for p in self.providers]

        if self.pseudoabsences is None:
            pseudoabsence_patches = 0
        else:
            pseudoabsence_item = self.pseudoabsence_items.iloc[index].to_dict()
            pseudoabsence_patches = [p[pseudoabsence_item] for p in self.providers]
        
        if self.test:
            labels = 0
        else:
            item_species = self.items.iloc[index][self.label_name]
            labels = 1 * np.isin(self.species_pred, item_species)

        return patches, pseudoabsence_patches, labels
