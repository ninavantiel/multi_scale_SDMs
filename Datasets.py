import numpy as np
import pandas as pd
import itertools
import os
import rasterio
import pyproj
import torch 
from torch.utils.data import Dataset

class RasterPatchProvider(object):
    def __init__(self, raster_path, normalize=True, nan_value=0, device="cpu"):
        with rasterio.open(raster_path) as src:
            self.meta = src.meta
            self.meta.update(count=src.count)
            self.data = src.read()
            self.nodata_value = src.nodatavals

            # iterate through all the layers
            for i in range(src.count):
                # replace the NoData values with np.nan
                self.data = self.data.astype(np.float)
                self.data[i] = np.where(self.data[i] == self.nodata_value[i], np.nan, self.data[i])
                if normalize:
                    self.data[i] = (self.data[i] - np.nanmean(self.data[i]))/np.nanstd(self.data[i])
                self.data[i] = np.where(np.isnan(self.data[i]), nan_value, self.data[i])
            
            # self.data = torch.tensor(self.data,  dtype=torch.float32).to(device)

            self.name = os.path.basename(os.path.splitext(raster_path)[0])
            self.nb_layers = src.count
            if self.nb_layers > 1:
                self.band_names = [self.name+'_'+str(i+1) for i in range(self.nb_layers)]
            else:   
                self.band_names = [self.name]
        
            self.x_min = src.bounds.left
            self.y_min = src.bounds.bottom
            self.x_resolution = src.res[0]
            self.y_resolution = src.res[1]
            self.n_rows = src.height
            self.n_cols = src.width

            self.crs = src.crs
            self.epsg = self.crs.to_epsg()
            if self.epsg != 4326:
                # create a pyproj transformer object to convert lat, lon to EPSG:32738
                self.transformer = pyproj.Transformer.from_crs("epsg:4326", self.epsg, always_xy=True)
            else:
                self.transformer = None

    def __len__(self):
        return self.nb_layers
    
    def __str__(self):
        return f"{self.name} - nb_layers = {self.nb_layers}"
        # result = '-' * 50 + '\n'
        # result += 'n_layers: ' + str(self.nb_layers) + '\n'
        # result += 'x_min: ' + str(self.x_min) + '\n'
        # result += 'y_min: ' + str(self.y_min) + '\n'
        # result += 'x_resolution: ' + str(self.x_resolution) + '\n'
        # result += 'y_resolution: ' + str(self.y_resolution) + '\n'
        # result += 'n_rows: ' + str(self.n_rows) + '\n'
        # result += 'n_cols: ' + str(self.n_cols) + '\n'
        # result += '-' * 50
        # return result
        

class MultipleRasterPatchProvider(object):
    def __init__(self, raster_folder, select=None, normalize=True, device="cpu"):
        files = os.listdir(raster_folder)
        if select:
            raster_paths = [raster_folder+r+'.tif' for r in select]
        else:
            rasters_paths = [raster_folder+f for f in files if f.endswith('.tif')]

        self.rasters_providers = [RasterPatchProvider(path, normalize=normalize) for path in rasters_paths]
        self.nb_layers = sum([len(raster) for raster in self.rasters_providers])
        self.band_names = list(itertools.chain.from_iterable([raster.band_names for raster in self.rasters_providers]))
    
    def __str__(self):
        return f"Rasters in folder: {self.nb_layers}\n{self.band_names}\n"

class PatchesDatasetMultiLabel(Dataset):
    def __init__(
            self, 
            occurrences_path,
            #providers,
            #transform=None,
            # target_transform=None,
            # id_name="glcID",
            # label_name="speciesId",
            # item_columns=['lat', 'lon', 'patchID'],
            device="cpu",
        ):

        occurrences = pd.read_csv(occurrences_path, sep=';', header='infer', low_memory=False)
        self.observation_ids = occurrences['glcID'].values
        self.items = occurrences[['lat', 'lon', 'patchID']]
        self.targets = occurrences['speciesId'].values



