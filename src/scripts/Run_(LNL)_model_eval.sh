#!/usr/bin/env bash

declare -a model_name='LNL_15x1'

declare -a in_length=2000

declare -a downsample_factor=5

declare -a unseen_data_labels=(
                              'sensor@[-3m]'
                              'sensor@[-2m]'
                              'sensor@[0m]'
                              'sensor@[5m]'
                              'sensor@[7m]'
                              'sensor@[16m]'
                              'sensor@[17m]'
                             )

declare -a seen_data_labels=(
                              'sensor@[-4m]'
                              'sensor@[-2m]'
                              'sensor@[2m]'
                              'sensor@[4m]'
                              'sensor@[6m]'
                              'sensor@[8m]'
                              'sensor@[10m]'
                             )

declare -a test_tdms_dir=(
                          'G:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/Leak/Test data/'
                          'G:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/NoLeak/Test data/'
                          'G:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/Leak/Test data/'
                          'G:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/NoLeak/Test data/'
                         )

declare -a fig_label=(
                      'Unseen-Leak'
                      'Unseen-NoLeak'
                      'Seen-Leak'
                      'Seen-NoLeak'
                     )

declare -a noleak_label=(0 0 0 0 0 0 0)

declare -a leak_label=(1 1 1 1 1 1 1)

declare -a model_possible_input=(0 1)

echo ------------------------------------------------------------------------------------------- Training
python dataset_experiment_2019_1_3_\(LNL\)_train_script.py

echo ------------------------------------------------------------------------------------------- Unseen leak
python dataset_experiment_2019_1_3_\(LNL\)_model_eval.py --model "${model_name}"\
                                                         --inlen ${in_length}\
                                                         --mpl ${model_possible_input[*]}\
                                                         --testdir "${test_tdms_dir[0]}"\
                                                         --dsf ${downsample_factor}\
                                                         --actlabel ${leak_label[*]}\
                                                         --inlabel ${unseen_data_labels[*]}\
                                                         --figname ${fig_label[0]}

echo ------------------------------------------------------------------------------------------- Unseen Noleak
python dataset_experiment_2019_1_3_\(LNL\)_model_eval.py --model "${model_name}"\
                                                         --inlen ${in_length}\
                                                         --mpl ${model_possible_input[*]}\
                                                         --testdir "${test_tdms_dir[1]}"\
                                                         --dsf ${downsample_factor}\
                                                         --actlabel ${noleak_label[*]}\
                                                         --inlabel ${unseen_data_labels[*]}\
                                                         --figname ${fig_label[1]}

echo ------------------------------------------------------------------------------------------- Seen leak
python dataset_experiment_2019_1_3_\(LNL\)_model_eval.py --model "${model_name}"\
                                                         --inlen ${in_length}\
                                                         --mpl ${model_possible_input[*]}\
                                                         --testdir "${test_tdms_dir[2]}"\
                                                         --dsf ${downsample_factor}\
                                                         --actlabel ${leak_label[*]}\
                                                         --inlabel ${seen_data_labels[*]}\
                                                         --figname ${fig_label[2]}

echo ------------------------------------------------------------------------------------------- Seen Noleak
python dataset_experiment_2019_1_3_\(LNL\)_model_eval.py --model "${model_name}"\
                                                         --inlen ${in_length}\
                                                         --mpl ${model_possible_input[*]}\
                                                         --testdir "${test_tdms_dir[3]}"\
                                                         --dsf ${downsample_factor}\
                                                         --actlabel ${noleak_label[*]}\
                                                         --inlabel ${seen_data_labels[*]}\
                                                         --figname ${fig_label[3]}

