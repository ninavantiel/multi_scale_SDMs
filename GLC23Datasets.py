# Author: Benjamin Deneu <benjamin.deneu@inria.fr>
#         Theo Larcher <theo.larcher@inria.fr>
#
# License: GPLv3
#
# Python version: 3.10.6

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from GLC23PatchesProviders import MetaPatchProvider
from GLC23TimeSeriesProviders import MetaTimeSeriesProvider

class PatchesDataset(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID']
        # ref_targets=None
    ):
        #print("PatchesDataset __init__")
        self.occurences = occurrences#Path(occurrences)
        self.base_providers = providers
        self.transform = transform
        self.target_transform = target_transform
        self.provider = MetaPatchProvider(self.base_providers, self.transform)

        df = occurrences#pd.read_csv(self.occurences, sep=";", header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values
        
        # if ref_targets is None:
        #     self.unique_sorted_targets = np.unique(np.sort(self.targets))
        # else:
        #     self.unique_sorted_targets = ref_targets

    def __len__(self):
        #print("PatchesDataset __init__")
        return len(self.observation_ids)

    def __getitem__(self, index):
        #print(f"PatchesDataset __getitem__ index={index}")

        item = self.items.iloc[index].to_dict()
        patch = self.provider[item]
        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patch).float(), target
    
    def plot_patch(self, index):
        item = self.items.iloc[index].to_dict()
        self.provider.plot_patch(item)


class PatchesDatasetMultiLabel(PatchesDataset):
    def __init__(self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID'],
        sorted_unique_targets=None
    ):
        #print("PatchesDatasetMultiLabel __init__")
        super().__init__(occurrences, providers, transform, target_transform, id_name, label_name, item_columns)
        if sorted_unique_targets is None:
            self.sorted_unique_targets = np.unique(np.sort(self.targets))
        else:
            self.sorted_unique_targets = sorted_unique_targets

    def __getitem__(self, index):
        # print(f"PatchesDatasetMultiLabel __getitem__ index={index}")
        item = self.items.iloc[index].to_dict()
        patchid_rows_i = self.items[self.items['patchID']==item['patchID']].index
        # self.targets_sorted = np.sort(self.targets)

        patch = self.provider[item]
        item_targets = np.zeros(len(self.sorted_unique_targets))
        for idx in patchid_rows_i:
            target = self.targets[idx]
            if self.target_transform:
                target = self.target_transform(target)
            item_targets[np.where(self.sorted_unique_targets==target)] = 1

        # print(index, len(item_targets), item_targets.sum(), item_targets[np.where(item_targets!=0)])
        item_targets = torch.from_numpy(item_targets)

        return torch.from_numpy(patch).float(), item_targets
'''
class PatchesDatasetOld(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID'],
    ):
        self.occurences = Path(occurrences)
        self.providers = providers
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(self.occurences, sep=";", header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        item = self.items.iloc[index].to_dict()

        patches = []
        for provider in self.providers:
            patches.append(provider[item])

        # Concatenate all patches into a single tensor
        if len(patches) == 1:
            patches = patches[0]
        else:
            patches = np.concatenate(patches, axis=0)

        if self.transform:
            patches = self.transform(patches)

        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patches).float(), target
'''
class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['timeSerieID']
    ):
        #print("TimeSeriesDataset __init__")
        self.occurences = Path(occurrences)
        self.base_providers = providers
        self.transform = transform
        self.target_transform = target_transform
        self.provider = MetaTimeSeriesProvider(self.base_providers, self.transform)

        df = pd.read_csv(self.occurences, sep=";", header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values
    
    def __len__(self):
        #print("TimeSeriesDataset __len__")
        return len(self.observation_ids)

    def __getitem__(self, index):
        #print(f"TimeSeriesDataset __getitem__ index={index}")
        item = self.items.iloc[index].to_dict()
        patch = self.provider[item]
        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patch).float(), target

    def plot_ts(self, index):
        item = self.items.iloc[index].to_dict()
        self.provider.plot_ts(item)