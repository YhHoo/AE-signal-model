# self lib
from src.utils.helpers import *

filename = 'lcp_index_1bar_near_segmentation4'
lcp_index_dir = 'C:/Users/YH/Desktop/hooyuheng.masterWork/LCP DATASET OCT 3 1BAR/'
filename_to_save = direct_to_dir(where='result') + filename + '.csv'
filename_list = [lcp_index_dir + filename + '_p{}.csv'.format(i) for i in range(5)]

for f in filename_list:
    print(f)


# jus to get the col name
df = pd.read_csv(filename_list[0], index_col=0)
column_name = df.columns.values

# collect data from all csv
data_arr_list = []
for f in filename_list:
    df = pd.read_csv(f, index_col=0)
    data_arr_list.append(df.values)

# concat all into an arr
all = np.concatenate(data_arr_list, axis=0)

# save to new csv
df_all = pd.DataFrame(data=all, columns=column_name)

df_all.to_csv(filename_to_save)

print('Saved --> ', filename_to_save)

