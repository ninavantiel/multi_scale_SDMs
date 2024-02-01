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
        pseudoabsences=None
    ):
        self.occurrences = Path(occurrences)
        self.label_name = label_name
        self.item_columns = item_columns
        self.pseudoabsences = pseudoabsences

        df = pd.read_csv(self.occurrences, sep=";", header='infer', low_memory=False)
        if species is None: 
            self.species = np.unique(df[label_name].values)
        else: 
            self.species = species
        self.items = pd.DataFrame(df.groupby(item_columns)[label_name].agg(list)).reset_index()

        n = self.items.shape[0]
        self.species_counts = pd.Series(
            [sps for sps_list in self.items[label_name] for sps in sps_list]
        ).value_counts().sort_index()
        self.species_weights = (n / self.species_counts).values

        if self.pseudoabsences is not None:
            self.pseudoabsence_items = pd.read_csv(self.pseudoabsences).sample(n)
            print('nb pseudoabsences = ', self.pseudoabsence_items.shape)

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
            return patch, labels
        
        else:
            pseudoabsence_item = self.pseudoabsence_items.iloc[index].to_dict()
            pseudoabsence_patch = self.provider[pseudoabsence_item]
            return patch, labels, pseudoabsence_patch

class MultiScalePatchesDatasetCooccurrences(PatchesDatasetCooccurrences):
    def __init__(
        self,
        occurrences,
        providers,
        species=None,
        label_name='speciesId',
        item_columns=['lat','lon','patchID','dayOfYear'],
    ):
        super().__init__(occurrences, providers, species,label_name, item_columns)
        self.providers = [MetaPatchProvider(p) for p in self.base_providers]

    def __getitem__(self, index):
        item = self.items.iloc[index][self.item_columns].to_dict()
        item_species = self.items[self.label_name]
        labels = 1 * np.isin(self.species, item_species)

        patch_list = [provider[item] for provider in self.providers]

        return patch_list, labels
    
        # add option to fetch pseudoabsences here later if necessary