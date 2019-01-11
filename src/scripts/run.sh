
#!/bin/bash

# FOR data_preparation_script
declare -a file_to_save=(
                         'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_leak_random_1.5bar_[0]_ds2.csv'
                         'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_noleak_random_1.5bar_[0]_ds2.csv'
                         'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds2.csv'
                         'C:/Users/YH/PycharmProjects/AE-signal-model/result/dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds2.csv'
                        )

declare -a folder_to_read=(
                           'E:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/Leak/Train & Val data/'
                           'E:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/NoLeak/Train & Val data/'
                           'E:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/Leak/Train & Val data/'
                           'E:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/NoLeak/Train & Val data/'
                          )

declare -a all_even_channel=(0 1 2 3 4 5 6)

#echo ------------------------------------------------------------------------------------------- leak [0]
#python test.py --fts "${file_to_save[0]}" --ftr "${folder_to_read[0]}" --cth ${all_even_channel[2]} --svs 6000 --dsf 10
#
#echo ------------------------------------------------------------------------------------------- noleak [0]
#python test.py --fts "${file_to_save[1]}" --ftr "${folder_to_read[1]}" --cth ${all_even_channel[2]} --svs 6000 --dsf 10
#
#echo ------------------------------------------------------------------------------------------- leak [-4,-2,2,4,6,8,10]
#python test.py --fts "${file_to_save[2]}" --ftr "${folder_to_read[2]}" --cth ${all_even_channel[*]} --svs 6000 --dsf 10
#
#echo ------------------------------------------------------------------------------------------- noleak [-4,-2,2,4,6,8,10]
#python test.py --fts "${file_to_save[3]}" --ftr "${folder_to_read[3]}" --cth ${all_even_channel[*]} --svs 6000 --dsf 10


#echo ------------------------------------------------------------------------------------------- leak [0]
#python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save[0]}" \
#                                                                              --ftr "${folder_to_read[0]}" \
#                                                                              --cth ${all_even_channel[2]} \
#                                                                              --svs 6000 \
#                                                                              --dsf 10
#
#echo ------------------------------------------------------------------------------------------- noleak [0]
#python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save[1]}" \
#                                                                              --ftr "${folder_to_read[1]}" \
#                                                                              --cth ${all_even_channel[2]} \
#                                                                              --svs 6000 \
#                                                                              --dsf 10
#
#echo ------------------------------------------------------------------------------------------- leak [-4,-2,2,4,6,8,10]
#python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save[2]}" \
#                                                                              --ftr "${folder_to_read[2]}" \
#                                                                              --cth ${all_even_channel[*]} \
#                                                                              --svs 6000 \
#                                                                              --dsf 10
#
echo ------------------------------------------------------------------------------------------- noleak [-4,-2,2,4,6,8,10]
python dataset_experiment_2018_10_3_\(LeakRandom\)_data_preparation_script.py --fts "${file_to_save[3]}" \
                                                                              --ftr "${folder_to_read[3]}" \
                                                                              --cth ${all_even_channel[*]} \
                                                                              --svs 6000 \
                                                                              --dsf 10