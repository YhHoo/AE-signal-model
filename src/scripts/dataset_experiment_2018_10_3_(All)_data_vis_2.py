

from src.utils.helpers import *


tdms_dir = 'G:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/NoLeak/Train & Val data/'
all_tdms_file = [(tdms_dir + f) for f in listdir(tdms_dir) if f.endswith('.tdms')]
print('total file to extract: ', len(all_tdms_file))

n_channel_data = read_single_tdms(all_tdms_file[40])
n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1, :]
print('Dim before visualize: ', n_channel_data.shape)

fig_time = plot_multiple_timeseries(input=n_channel_data,
                                    subplot_titles=['sensor@[-4m]',  # the channels' dist of the input data
                                                    'sensor@[-2m]',
                                                    'sensor@[2m]',
                                                    'sensor@[4m]',
                                                    'sensor@[6m]',
                                                    'sensor@[8m]',
                                                    'sensor@[10m]'],
                                    main_title='Time plot')

plt.show()