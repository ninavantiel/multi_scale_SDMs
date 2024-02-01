import numpy as np
import pandas as pd

from data.PatchesProviders import RasterPatchProvider, MultipleRasterPatchProvider

def generate_pseudoabsence_locations(providers, bounds, out_path):
    n = 0
    try:
        pseudoabsence_locations = np.zeros(providers[0].rasters_providers[0].data.shape[1:])
        for p in providers:
            for subp in p.rasters_providers:
                pseudoabsence_locations += np.where(subp.data.sum(axis=0) == 0, 0, 1)
                n += 1
    except:
        pseudoabsence_locations = np.zeros(providers.rasters_providers[0].data.shape[1:])
        for p in providers.rasters_providers:
            pseudoabsence_locations += np.where(p.data.sum(axis=0) == 0, 0, 1)
            n += 1

    df = np.argwhere(pseudoabsence_locations == n)
    df = pd.DataFrame(df, columns=['x','y'])

    try:
        p = providers[0].rasters_providers[0]
    except:
        p = providers.rasters_providers[0]

    df['lat'] = p.y_min + p.y_resolution * (p.n_rows - df['x'])
    df['lon'] = p.x_min + p.x_resolution * df['y']

    df = df[(df.lat >= bounds['lat_min']) & (df.lat <= bounds['lat_max']) &
            (df.lon >= bounds['lon_min']) & (df.lon <= bounds['lon_max'])]
    
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    datadir = 'data/full_data/'
    po_path = datadir+'Presence_only_occurrences/Presences_only_train_sampled_100_percent_min_1_occurrences.csv'
    bioclim_dir = datadir+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/'
    soil_dir = datadir+'EnvironmentalRasters/Soilgrids/'

    out_path = datadir+'Presence_only_occurrences/Pseudoabsence_locations_bioclim_soil.csv'

    patch_size = 1
    flatten = True
    p_bioclim = MultipleRasterPatchProvider(bioclim_dir, size=patch_size, flatten=flatten) 
    p_soil = MultipleRasterPatchProvider(soil_dir, size=patch_size, flatten=flatten) 

    df = pd.read_csv(po_path, sep=";", header='infer', low_memory=False)
    bounds = {'lat_min': df.lat.min(), 'lat_max': df.lat.max(),
              'lon_min': df.lon.min(), 'lon_max': df.lon.max()}

    generate_pseudoabsence_locations((p_bioclim, p_soil), bounds, out_path)