from abc import abstractmethod
import itertools
import numpy as np
import os
import rasterio
import pyproj

class PatchProvider(object):
    def __init__(self, size, normalize) -> None:
        self.patch_size = size
        self.normalize = normalize
        self.nb_layers = 0

    @abstractmethod
    def __getitem__(self, item):
        pass
    
    def __repr__(self):
        return self.__str__()
    
    @abstractmethod
    def __str__(self):
        pass
    
    def __len__(self):
        return self.nb_layers
    
    # def plot_patch(self, item):
    # TO DO

class MetaPatchProvider(PatchProvider):
    def __init__(self, providers, one_provider):
        self.providers = providers
        self.one_provider = one_provider
        
        if one_provider:
            self.nb_layers = len(self.providers)
            self.band_names = self.providers.bands_names
        else:
            self.nb_layers = sum([len(provider) for provider in self.providers])
            self.bands_names = list(itertools.chain.from_iterable([provider.bands_names for provider in self.providers]))            
            
    def __getitem__(self, item):
        if self.one_provider: 
            patch = self.providers[item]
        else:
            patch = np.concatenate([provider[item] for provider in self.providers])
        return patch
    
    def __str__(self):
        result = 'Providers:\n'
        for provider in self.providers:
            result += str(provider)
            result += '\n'
        return result
    
class RasterPatchProvider(PatchProvider):
    def __init__(self, raster_path, size=128, flatten=False, normalize=True, fill_zero_if_error=False, nan_value=0):
        super().__init__(size, normalize)
        self.raster_path = raster_path
        self.flatten = flatten
        self.fill_zero_if_error = fill_zero_if_error
        self.transformer = None
        self.name = os.path.basename(os.path.splitext(raster_path)[0])
        self.normalize = normalize
        self.nan_value = nan_value

        # open the tif file with rasterio
        with rasterio.open(self.raster_path) as src:
            # read the data from the raster
            self.data = src.read()

            self.nodata_value = src.nodatavals
            self.nb_layers = src.count
            self.x_min = src.bounds.left # minimum longitude
            self.y_min = src.bounds.bottom # minimum latitude
            self.x_resolution = src.res[0]
            self.y_resolution = src.res[1]
            self.n_rows = src.height
            self.n_cols = src.width
            self.crs = src.crs

            # iterate through all the layers
            for i in range(src.count):
                # replace the NoData values with np.nan
                self.data = self.data.astype(np.float)
                self.data[i] = np.where(self.data[i] == self.nodata_value[i], np.nan, self.data[i])
                # normalize layer
                if self.normalize:
                    self.data[i] = (self.data[i] - np.nanmean(self.data[i]))/np.nanstd(self.data[i])
                # replace np.nan entries with nan_value (default is 0)
                self.data[i] = np.where(np.isnan(self.data[i]), self.nan_value, self.data[i])

        if self.nb_layers > 1:
            self.bands_names = [self.name+'_'+str(i+1) for i in range(self.nb_layers)]
        else:
            self.bands_names = [self.name]
        
        self.epsg = self.crs.to_epsg()
        if self.epsg != 4326:
            # create a pyproj transformer object to convert lat, lon to EPSG:32738
            self.transformer = pyproj.Transformer.from_crs("epsg:4326", self.epsg, always_xy=True)

    def __getitem__(self, item):
        """
        :param item: dictionary that needs to contains at least the keys latitude and longitude ({'lat': lat, 'lon':lon})
        :return: return the environmental tensor or vector (size>1 or size=1)
        """
        # convert the lat, lon coordinates to EPSG:32738
        if self.transformer:
            lon, lat = self.transformer.transform(item['lon'], item['lat'])
        else:
            lon, lat = (item['lon'], item['lat'])
        
        # calculate the x, y coordinate of the point of interest
        x = int(self.n_rows - (lat - self.y_min) / self.y_resolution)
        y = int((lon - self.x_min) / self.x_resolution)

        if self.patch_size == 1:
            patch_data = [self.data[i, x, y] for i in range(self.nb_layers)]
        else:
            patch_data = [self.data[i, x - (self.patch_size // 2): x + (self.patch_size // 2), y - (self.patch_size // 2): y + (self.patch_size // 2)] for i in range(self.nb_layers)]

        tensor = np.concatenate([patch[np.newaxis] for patch in patch_data])
        if self.fill_zero_if_error and tensor.shape != (self.nb_layers, self.patch_size, self.patch_size):
            tensor = np.zeros((self.nb_layers, self.patch_size, self.patch_size))
        if self.flatten:
            tensor = tensor.flatten()
        return tensor

    def __str__(self):
        result = '-' * 50 + '\n'
        result = 'band_names: ' + str(self.bands_names) + '\n'
        result += 'n_layers: ' + str(self.nb_layers) + '\n'
        result += 'x_min: ' + str(self.x_min) + '\n'
        result += 'y_min: ' + str(self.y_min) + '\n'
        result += 'x_resolution: ' + str(self.x_resolution) + '\n'
        result += 'y_resolution: ' + str(self.y_resolution) + '\n'
        result += 'n_rows: ' + str(self.n_rows) + '\n'
        result += 'n_cols: ' + str(self.n_cols) + '\n'
        result += '-' * 50
        return result

class MultipleRasterPatchProvider(PatchProvider):
    def __init__(self, rasters_folder, select=None, size=128, flatten=False, normalize=True, fill_zero_if_error=False):
        files = os.listdir(rasters_folder)
        if select:
            rasters_paths = [r+'.tif' for r in select]
        else:
            rasters_paths = [f for f in files if f.endswith('.tif')]
        self.rasters_providers = [RasterPatchProvider(
            rasters_folder+path, size=size, flatten=flatten, normalize=normalize, fill_zero_if_error=fill_zero_if_error
        ) for path in rasters_paths]
        self.nb_layers = sum([len(raster) for raster in self.rasters_providers])
        self.bands_names = list(itertools.chain.from_iterable([raster.bands_names for raster in self.rasters_providers]))
    
    def __getitem__(self, item):
        return np.concatenate([raster[item] for raster in self.rasters_providers])
    
    def __str__(self):
        result = 'Rasters in folder:\n'
        for raster in self.rasters_providers:
            result += str(raster)
            result += '\n'
        return result