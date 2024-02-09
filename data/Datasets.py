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
        item_columns=['lat','lon','patchID','dayOfYear'],
        pseudoabsences=None,
        n_low_occ=50
    ):
        self.occurrences = Path(occurrences)
        self.label_name = label_name
        self.item_columns = item_columns
        self.pseudoabsences = pseudoabsences
        self.n_low_occ = n_low_occ

        df = pd.read_csv(self.occurrences, sep=";", header='infer', low_memory=False)
        if species is None: 
            self.species = np.unique(df[label_name].values)
        else: 
            self.species = species
        self.items = pd.DataFrame(df.groupby(item_columns)[label_name].agg(list)).reset_index()

        self.n_items = self.items.shape[0]
        self.n_species = len(self.species)
        print(f"nb items = {self.n_items}\nnb species = {self.n_species}")

        self.species_counts = pd.Series(
            [sps for sps_list in self.items[label_name] for sps in sps_list]
        ).value_counts().sort_index()
        self.species_weights = (self.n_items / self.species_counts).values

        self.low_occ_species = self.species_counts[self.species_counts <= self.n_low_occ].index
        self.low_occ_species_idx = np.where(np.isin(self.species, self.low_occ_species))[0]
        print(f"nb of species with less than {self.n_low_occ} occurrences = {len(self.low_occ_species_idx)}")

        if self.pseudoabsences is not None:
            self.pseudoabsence_items = pd.read_csv(self.pseudoabsences).sample(self.n_items)
            print(f"nb pseudoabsences = {self.pseudoabsence_items.shape[0]}")

        self.base_providers = providers
        self.provider = MetaPatchProvider(self.base_providers)

    def __len__(self):
        return self.items.shape[0]

    def __getitem__(self, index):
        item = self.items.iloc[index][self.item_columns].to_dict()
        item_species = self.items.iloc[index][self.label_name]
        labels = 1 * np.isin(self.species, item_species)
        patch = self.provider[item]

        if self.pseudoabsences is None:
            pseudoabsence_patch = None        
        else:
            pseudoabsence_item = self.pseudoabsence_items.iloc[index].to_dict()
            pseudoabsence_patch = self.provider[pseudoabsence_item]
        
        return patch, labels, pseudoabsence_patch

class MultiScalePatchesDatasetCooccurrences(PatchesDatasetCooccurrences):
    def __init__(
        self,
        occurrences,
        providersA,
        providersB,
        species=None,
        label_name='speciesId',
        item_columns=['lat','lon','patchID','dayOfYear'],
        pseudoabsences=None
    ):
        super().__init__(occurrences, providersA, species,label_name, item_columns, pseudoabsences)
        
        self.base_providersA = providersA
        self.providersA = MetaPatchProvider(self.base_providersA)
        self.base_providersB = providersB
        self.providersB = MetaPatchProvider(self.base_providersB)

    def __getitem__(self, index):
        item = self.items.iloc[index][self.item_columns].to_dict()
        item_species = self.items.iloc[index][self.label_name]
        labels = 1 * np.isin(self.species, item_species)

        patchA = self.providersA[item]
        patchB = self.providersB[item]
        
        if self.pseudoabsences is None:
            pseudoabsence_patch_list = None
        else:
            pseudoabsence_item = self.pseudoabsence_items.iloc[index].to_dict()
            pseudoabsence_patch_list = [provider[pseudoabsence_item] for provider in self.providers]
        
        return patchA, patchB, labels, pseudoabsence_patch_list
