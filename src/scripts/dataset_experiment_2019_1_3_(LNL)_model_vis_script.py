import time
from src.utils.helpers import *
from src.utils.dsp_tools import *

# No Normalized & Downsampled to fs=25kHz (2000 points per sample) --------------------------------------------
leak_p1 = 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds4.csv'
leak_p2 = 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[0]_ds4.csv'
leak_p3 = 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds4.csv'

noleak_p1 = 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds4.csv'
noleak_p2 = 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_noleak_random_1.5bar_[0]_ds4.csv'
noleak_p3 = 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds4.csv'

df_leak_rand_p1 = pd.read_csv(leak_p1)
df_leak_rand_p2 = pd.read_csv(leak_p2)
df_leak_rand_p3 = pd.read_csv(leak_p3)
df_noleak_rand_p1 = pd.read_csv(noleak_p1)
df_noleak_rand_p2 = pd.read_csv(noleak_p2)
df_noleak_rand_p3 = pd.read_csv(noleak_p3)


# NOLEAK ---------------------------------------------------------------------------------------------------------------
# [-4,-2,2,4,6,8,10]
noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
# shuffle
noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

# [0]
noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
# shuffle
noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

# [-3,5,7,16,17]
noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
# shuffle
noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]

# link all
noleak_all = np.concatenate((noleak_data_p1[:500, :], noleak_data_p2[:500, :], noleak_data_p3[:500, :]), axis=0)

# LEAK -----------------------------------------------------------------------------------------------------------------
# [-4,-2,2,4,6,8,10]
leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
# shuffle
leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]

# [0]
leak_data_p2 = df_leak_rand_p2.values[:, :-1]
# shuffle
leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]

# [-3,5,7,16,17]
leak_data_p3 = df_leak_rand_p3.values[:, :-1]
# shuffle
leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]

# link all
leak_all = np.concatenate((leak_data_p1[:500, :], leak_data_p2[:500, :], leak_data_p3[:500, :]), axis=0)


print('TOTAL NOLEAK:', noleak_all.shape)
print('TOTAL LEAK:', leak_all.shape)

lcp_model = load_model(model_name='LNL_44x6')
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')

activation = get_activations(lcp_model,
                             model_inputs=[noleak_all[0].reshape((6000, 1))],
                             print_shape_only=True)


