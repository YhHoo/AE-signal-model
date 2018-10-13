# self lib
from src.utils.helpers import *

# CONFIG ---------------------------------------------------------------------------------------------------------------
lcp_index_dir = 'C:/Users/YH/Desktop/hooyuheng.masterWork/LCP DATASET OCT 3 1BAR/'
filename_to_save = direct_to_dir(where='result') + 'lcp_index_1bar_near_segmentation4_p0_8.csv'
filename_to_read_list = [(lcp_index_dir + f) for f in listdir(lcp_index_dir) if f.endswith('.csv')]

# to confirm all files involved is true
for f in filename_to_read_list:
    print(f)

# READ AND APPEND ------------------------------------------------------------------------------------------------------
# jus to get the col name
df = pd.read_csv(filename_to_read_list[0], index_col=0)
column_name = df.columns.values

# collect data from all csv
data_arr_list = []
for f in filename_to_read_list:
    df = pd.read_csv(f, index_col=0)
    data_arr_list.append(df.values)

# concat all into an arr
all = np.concatenate(data_arr_list, axis=0)

# save to new csv
df_all = pd.DataFrame(data=all, columns=column_name)

df_all.to_csv(filename_to_save)

print('Saved --> ', filename_to_save)

