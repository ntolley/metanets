import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMapsMasker


data_dir = '/users/ntolley/scratch/metanets_data/'

atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

# Loading the functional datasets
data = datasets.fetch_development_fmri(data_dir=data_dir)

masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)

n_subjects = len(data.func)

df_list = list()
for subj_idx in range(n_subjects):
    time_series = masker.fit_transform(data.func[subj_idx], confounds=data.confounds)
    df_temp = pd.DataFrame(time_series)
    df_temp['subj'] = np.repeat(subj_idx, len(df_temp))

    df_list.append(df_temp)

df = pd.concat(df_list)

df = pd.read_pickle('../data/developmental_df.pkl')