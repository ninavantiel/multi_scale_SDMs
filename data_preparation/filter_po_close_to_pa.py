import numpy as np
import pandas as pd
import geopandas as gpd

datadir = 'data/full_data/'
po_path = datadir+'Presence_only_occurrences/Presences_only_train.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

min_dist_km = 50
out_path = datadir+f"Presence_only_occurrences/Presences_only_train_min_{min_dist_km}_from_pa.csv"

pa_df = pd.read_csv(pa_path, sep=";", header='infer', low_memory=False)
po_df = pd.read_csv(po_path, sep=";", header='infer', low_memory=False)
print(f"INITIAL SHAPE OF PO DATASET: {po_df.shape}, NB SPECIES={len(np.unique(po_df['speciesId'].values))}")

pa_gdf = gpd.GeoDataFrame(pa_df, geometry=gpd.points_from_xy(pa_df.x_EPSG3035, pa_df.y_EPSG3035, crs="EPSG:3035"))
po_gdf = gpd.GeoDataFrame(po_df, geometry=gpd.points_from_xy(po_df.x_EPSG3035, po_df.y_EPSG3035, crs="EPSG:3035"))

po_gdf['dist_to_pa'] = po_gdf.apply(lambda p: pa_gdf.distance(p.geometry).min() / 1e3, axis=1)
po_gdf_filtered = po_gdf[po_gdf['dist_to_pa'] >= min_dist_km]
print(f"FILTER OUT POINTS LESS THAN {min_dist_km} KM FROM PA POINTS-> SHAPE: {po_gdf_filtered.shape}, NB SPECIES={len(np.unique(po_gdf_filtered['speciesId'].values))}")

po_gdf_filtered.drop(columns=['geometry']).to_csv(out_path, sep=';')