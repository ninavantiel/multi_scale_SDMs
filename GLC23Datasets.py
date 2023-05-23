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
        return len(self.observation_ids)

    def __getitem__(self, index):
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

# class PatchesDatasetMultiLabel_persite(PatchesDataset):
#     def __init__(self,
#         occurrences,
#         providers,
#         transform=None,
#         target_transform=None,
#         id_name="glcID",
#         label_name="speciesId",
#         item_columns=['lat', 'lon', 'patchID'],
#         site_columns=['patchID','dayOfYear'],
#         sorted_unique_targets=None
#     ):
#         #print("PatchesDatasetMultiLabel __init__")
#         super().__init__(occurrences, providers, transform, target_transform, id_name, label_name, item_columns)
#         if sorted_unique_targets is None:
#             self.sorted_unique_targets = np.unique(np.sort(self.targets))
#         else:
#             self.sorted_unique_targets = sorted_unique_targets

#         self.unique_sites = occurrences[site_columns].drop_duplicates().reset_index(drop=True)

#     def __len__(self):
#         return self.unique_sites.shape[0]

#     def __getitem__(self, index):
#         # print(f"PatchesDatasetMultiLabel __getitem__ index={index}")
#         site = self.unique_sites.iloc[index].to_dict()
#         patchid_rows_i = self.items[self.items['patchID']==site['patchID']].index
#         # self.targets_sorted = np.sort(self.targets)

#         patch = self.provider[item]
#         item_targets = np.zeros(len(self.sorted_unique_targets))
#         for idx in patchid_rows_i:
#             target = self.targets[idx]
#             if self.target_transform:
#                 target = self.target_transform(target)
#             item_targets[np.where(self.sorted_unique_targets==target)] = 1

#         # print(index, len(item_targets), item_targets.sum(), item_targets[np.where(item_targets!=0)])
#         item_targets = torch.from_numpy(item_targets)

#         return torch.from_numpy(patch).float(), item_targets


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


class TabularDataset(Dataset):
    def __init__(
            self,
            occurrences,
            env,
            covariates,
            sorted_unique_targets=None
    ):
        self.occurrences = occurrences
        self.env = env
        self.covariates = covariates

        if sorted_unique_targets is None:
            self.sorted_unique_targets = occurrences.speciesId.sort_values().unique()
        else:
            self.sorted_unique_targets = sorted_unique_targets

        self.env_norm = (env[covariates]-env[covariates].mean()) / env[covariates].std()

    def __len__(self):
        return self.env.shape[0]

    def __getitem__(self, index):
        item_env = self.env_norm.iloc[index].values
        
        item = self.env.iloc[index][['patchID', 'dayOfYear']].to_dict()
        labels = self.occurrences[(self.occurrences['patchID'] == item['patchID']) & (self.occurrences['dayOfYear'] == item['dayOfYear'])].speciesId.values
        item_targets = 1* np.isin(self.sorted_unique_targets, labels)

        return item_env, item_targets


