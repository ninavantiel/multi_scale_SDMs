from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

from data.PatchesProviders import MetaPatchProvider

class PatchesDataset(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        # one_provider=False,
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
        self.provider = MetaPatchProvider(self.base_providers)#, one_provider)

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
    ):
        self.occurrences = Path(occurrences)
        self.label_name = label_name
        self.item_columns = item_columns

        df = pd.read_csv(self.occurrences, sep=";", header='infer', low_memory=False)
        if species is None: 
            self.species = np.unique(df[label_name].values)
        else: 
            self.species = species
        self.items = pd.DataFrame(df.groupby(item_columns)[label_name].agg(list)).reset_index()

        self.base_providers = providers
        self.provider = MetaPatchProvider(self.base_providers)

    def __len__(self):
        return self.items.shape[0]

    def __getitem__(self, index):
        item = self.items.iloc[index][self.item_columns].to_dict()
        item_species = self.items[self.label_name]
        labels = 1 * np.isin(self.species, item_species)

        patch = self.provider[item]

        return patch, labels

class MultiScalePatchesDatasetCooccurrences(Dataset):
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
        self.items = pd.DataFrame(df.groupby(item_columns)[label_name].agg(list)).reset_index()

        self.base_providers = providers
        self.providers = [MetaPatchProvider(p) for p in self.base_providers]

    def __len__(self):
        return self.items.shape[0]

    def __getitem__(self, index):
        item = self.items.iloc[index][self.item_columns].to_dict()
        item_species = self.items[self.label_name]
        labels = 1 * np.isin(self.species, item_species)

        patch_list = [provider[item] for provider in self.providers]

        return patch_list, labels