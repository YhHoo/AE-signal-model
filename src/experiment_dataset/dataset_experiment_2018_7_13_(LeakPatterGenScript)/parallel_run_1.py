import time
from os import listdir
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

# sharing the total files
folder_path = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/'
all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
all_file_path = all_file_path[0:19]
print('FILE TO PROCESSED:')
for f in all_file_path:
    print(f)


ae_data = AcousticEmissionDataSet_13_7_2018(drive='F')
time_start = time.time()
ae_data.generate_leak_1bar_in_cwt_xcor_maxpoints_vector(saved_filename='sliced_xcor_2_p1', file_to_process=all_file_path)
print('Time taken: {:.4f}s'.format(time.time()-time_start))
