import rasterio
import numpy as np
import os

def pre_process_raster(raster_path, out_dir, normalize=True, nan_value=0):
    with rasterio.open(raster_path) as src:
        data = src.read()
        data = data.astype(float)
        profile = src.profile
        nodata_value = src.nodatavals
        name = os.path.basename(os.path.splitext(raster_path)[0])
        
        # iterate through all the layers
        for i in range(src.count):
            # replace the NoData values with np.nan
            data[i] = np.where(data[i] == nodata_value[i], np.nan, data[i])
            # normalize layer
            if normalize:
                data[i] = (data[i] - np.nanmean(data[i]))/np.nanstd(data[i])
            # replace np.nan entries with nan_value (default is 0)
            data[i] = np.where(np.isnan(data[i]), nan_value, data[i])

        with rasterio.open(out_dir+name+'_preprocessed.tif', 'w', **profile) as dst:
             dst.write(data)#src.count) # the numer one is the number of bands
        
if __name__ == "__main__":
    datadir = 'data/full_data/EnvironmentalRasters/'

    rasters_folder = datadir+'Climate/BioClimatic_Average_1981-2010/'
    out_folder = datadir+'preprocessed/BioClimatic_Average_1981-2010/'
    
    if not os.path.exists(out_folder): os.makedirs(out_folder)
    files = os.listdir(rasters_folder)
    rasters_paths = [f for f in files if f.endswith('.tif')]
    for f in files:
        print(f)
        if f.endswith('tif'): 
            pre_process_raster(rasters_folder+f, out_folder)
        break


