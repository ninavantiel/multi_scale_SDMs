import pandas as pd
import numpy as np

datadir = 'data/full_data/'
po_path = datadir+'Presence_only_occurrences/Presences_only_train.csv'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

out_path = datadir+f"Presence_only_occurrences/Presences_only_train_sampled_species_in_val.csv"

pa_df = pd.read_csv(pa_path, sep=";", header='infer', low_memory=False)
po_df = pd.read_csv(po_path, sep=";", header='infer', low_memory=False)

# filter out species that do not occur in PA dataset
pa_species = np.unique(pa_df['speciesId'].values)
po_df_filter = po_df[po_df['speciesId'].isin(pa_species)]
po_df_filter.to_csv(out_path, sep=';')


