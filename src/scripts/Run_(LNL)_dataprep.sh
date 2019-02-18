
#!/bin/bash


# FOR # FOR SEEN DATASETS -4,-2,2,4,6,8,10 ################################################################################

# FOR data_preparation_script
declare -a file_to_save=(
                         'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_leak_random_1.5bar_[0]_ds6.csv'
                         'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_noleak_random_1.5bar_[0]_ds6.csv'
                         'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds6.csv'
                         'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds6.csv'
                        )

declare -a folder_to_read=(
                           'G:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/Leak/Train & Val data/'
                           'G:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/NoLeak/Train & Val data/'
                           'G:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/Leak/Train & Val data/'
                           'G:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/NoLeak/Train & Val data/'
                          )

declare -a all_even_channel=(0 1 2 3 4 5 6)


echo ------------------------------------------------------------------------------------------- leak [0]
python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save[0]}" \
                                                                              --ftr "${folder_to_read[0]}" \
                                                                              --cth ${all_even_channel[2]} \
                                                                              --svs 2000 \
                                                                              --dsf 1000

echo ------------------------------------------------------------------------------------------- noleak [0]
python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save[1]}" \
                                                                              --ftr "${folder_to_read[1]}" \
                                                                              --cth ${all_even_channel[2]} \
                                                                              --svs 2000 \
                                                                              --dsf 1000

echo ------------------------------------------------------------------------------------------- leak [-4,-2,2,4,6,8,10]
python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save[2]}" \
                                                                              --ftr "${folder_to_read[2]}" \
                                                                              --cth ${all_even_channel[*]} \
                                                                              --svs 2000 \
                                                                              --dsf 1000
#
echo ------------------------------------------------------------------------------------------- noleak [-4,-2,2,4,6,8,10]
python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save[3]}" \
                                                                              --ftr "${folder_to_read[3]}" \
                                                                              --cth ${all_even_channel[*]} \
                                                                              --svs 2000 \
                                                                              --dsf 1000

# FOR UNSEEN DATASETS -3,-2,0,5,7,16,17 ################################################################################


# FOR data_preparation_script
declare -a file_to_save_2=(
                           'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds6.csv'
                           'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds6.csv'
                          )

declare -a folder_to_read_2=(
                             'G:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/Leak/Train & Val data/'
                             'G:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/NoLeak/Train & Val data/'
                            )

declare -a all_even_channel_2=(0 3 4 5 6)


echo ------------------------------------------------------------------------------------------- leak [-3,5,7,16,17]
python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save_2[0]}" \
                                                                              --ftr "${folder_to_read_2[0]}" \
                                                                              --cth ${all_even_channel_2[*]} \
                                                                              --svs 2000 \
                                                                              --dsf 1000

echo ------------------------------------------------------------------------------------------- noleak [-3,5,7,16,17]
python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save_2[1]}" \
                                                                              --ftr "${folder_to_read_2[1]}" \
                                                                              --cth ${all_even_channel_2[*]} \
                                                                              --svs 2000 \
                                                                              --dsf 1000
