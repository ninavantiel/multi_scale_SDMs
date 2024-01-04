from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

from data.PatchesProviders import MetaPatchProvider


class BioclimDataset(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        one_provider=False,
        species=None,
        id_name='glcID',
        label_name='speciesId',
        item_columns=['lat','lon','patchID','dayOfYear'],
    ):
        self.occurrences = Path(occurrences)
        self.label_name = label_name
        self.item_columns = item_columns

        df = pd.read_csv(self.occurrences, sep=";", header='infer', low_memory=False)
        # self.observation_ids = df[id_name].values
        self.items = df[item_columns + [label_name]]
        
        if species is None: 
            self.species = np.unique(df[label_name].values)
        else: 
            self.species = species

        self.base_providers = providers
        self.provider = MetaPatchProvider(self.base_providers, one_provider)

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        item = self.items.iloc[index][self.item_columns].to_dict()
        print(item)

        item_species = self.items.query(
            ' and '.join([f'{k} == {v}' for k, v in item.items()])
        )[self.label_name].values
        labels = 1 * np.isin(self.species, item_species)

        patch = self.provider[item]

        return patch, labels
