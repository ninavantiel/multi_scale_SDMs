import pandas as pd
from sklearn.model_selection import train_test_split

data_path = 'data/full_data/'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train.csv'
presence_only_sampled_path = data_path+'Presence_only_occurrences/Presences_only_train_sampled.csv'
presence_only_sampled_100_path = data_path+'Presence_only_occurrences/Presences_only_train_sampled_100.csv'

df = pd.read_csv(presence_only_path, sep=';')
print(f"Shape of presence only data: ", df.shape)

# thresh = 100
# frac = 0.5
# species_counts = df.groupby('speciesId').glcID.count().sort_values()
# df_no_sample = df[df['speciesId'].isin(species_counts[species_counts <= thresh].index)]
# df_sample = df[df['speciesId'].isin(species_counts[species_counts > thresh].index)].groupby(
#     'speciesId', group_keys=False).apply(lambda x: x.sample(frac=frac))
# df_sampled = pd.concat([df_no_sample, df_sample])
# print(f"Shape of *sampled* presence only data: ", df_sampled.shape)

# df_sampled.to_csv(presence_only_sampled_path, sep=';')
# print(f"Sampled presence only data saved!")

thresh = 100
species_counts = df.groupby('speciesId').glcID.count().sort_values()
df_no_sample = df[df['speciesId'].isin(species_counts[species_counts <= thresh].index)]
df_sample = df[df['speciesId'].isin(species_counts[species_counts > thresh].index)].groupby(
    'speciesId', group_keys=False).apply(lambda x: x.sample(thresh))
df_sampled = pd.concat([df_no_sample, df_sample])
print(f"Shape of *sampled* presence only data: ", df_sampled.shape)

df_sampled.to_csv(presence_only_sampled_100_path, sep=';')
print(f"Sampled presence only data saved!")


