import pandas as pd
import numpy as np

datadir = 'data/full_data/'
po_path = datadir+'Presence_only_occurrences/Presences_only_train.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

f = 0.1 #0.25
n_min = 100 #10
out_path = datadir+f"Presence_only_occurrences/Presences_only_train_sampled_{int(f*100)}_percent_min_{n_min}_occurrences.csv"

pa_df = pd.read_csv(pa_path, sep=";", header='infer', low_memory=False)
po_df = pd.read_csv(po_path, sep=";", header='infer', low_memory=False)
print(f"INITIAL SHAPE OF PO DATASET: {po_df.shape}, NB SPECIES={len(np.unique(po_df['speciesId'].values))}")

# filter out species that do not occur in PA dataset
pa_species = np.unique(pa_df['speciesId'].values)
po_df_filter = po_df[po_df['speciesId'].isin(pa_species)]
print(f"FILTER OUT SPECIES THAT ARE NOT IN PA -> SHAPE: {po_df_filter.shape}, NB SPECIES={len(np.unique(po_df_filter['speciesId'].values))}")

# sample 25% (f) of PO data
po_df_filter = po_df_filter.sample(frac=f)
print(f"SAMPLE {int(f*100)}% OF DATA -> SHAPE: {po_df_filter.shape}, NB SPECIES={len(np.unique(po_df_filter['speciesId'].values))}")

# filter out species with less than 10 (n_min) occurrences
sp_counts = po_df_filter.groupby('speciesId').glcID.count()
po_df_filter = po_df_filter[po_df_filter['speciesId'].isin(sp_counts[sp_counts >= n_min].index)]
print(f"FILTER OUT SPECIES WITH LESS THAN {n_min} OCCURRENCES-> SHAPE: {po_df_filter.shape}, NB SPECIES={len(np.unique(po_df_filter['speciesId'].values))}")

po_df_filter.to_csv(out_path, sep=';')


