import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

datadir = 'data/full_data/'
pa_path = datadir+'Presence_Absence_surveys/Presences_Absences_train.csv'

f_test = 0.2
out_path_val = datadir+f"Presence_Absence_surveys/Presences_Absences_val_{int((1-f_test)*100)}_percent.csv"
out_path_test = datadir+f"Presence_Absence_surveys/Presences_Absences_test_{int(f_test*100)}_percent.csv"

pa_df = pd.read_csv(pa_path, sep=";", header='infer', low_memory=False)
print(f"INITIAL SHAPE OF PA DATASET: {pa_df.shape}")

# split into two dataframes 
val, test = train_test_split(pa_df, test_size=f_test)
print(f"SHAPE OF VALIDATION PA DATASET: {val.shape}")
print(f"SHAPE OF TEST PA DATASET: {test.shape}")

val.to_csv(out_path_val, sep=';')
test.to_csv(out_path_test, sep=';')