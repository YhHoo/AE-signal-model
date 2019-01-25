#!/usr/bin/env bash

declare -a model_name='LNL_26x1'  # **

declare -a result_save_filename='C:/Users/YH/PycharmProjects/AE-signal-model/result/LNL_26x1_result.txt'  # **

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
#                                                                                    ********
declare -a pred_result_save_dir=(
                                 'G:/Experiment_3_1_2019/LNL_model_Evaluation_Result/LNL_26x1 evaluate with Experiment_21_12_2018_8Ch_-3,-2,0,5,7,16,17_1.5 bar_Leak_Test data/'
                                 'G:/Experiment_3_1_2019/LNL_model_Evaluation_Result/LNL_26x1 evaluate with Experiment_21_12_2018_8Ch_-3,-2,0,5,7,16,17_1.5 bar_NoLeak_Test data/'
                                 'G:/Experiment_3_1_2019/LNL_model_Evaluation_Result/LNL_26x1 evaluate with Experiment_21_12_2018_8Ch_-4,-2,2,4,6,8,10_1.5 bar_Leak_Test data/'
                                 'G:/Experiment_3_1_2019/LNL_model_Evaluation_Result/LNL_26x1 evaluate with Experiment_21_12_2018_8Ch_-4,-2,2,4,6,8,10_1.5 bar_NoLeak_Test data/'
                                )
#                                                                                                                                           **>                                 ********
declare -a cm_save_dir=(
                        'C:/Users/YH/Desktop/hooyuheng.master/MASTER_PAPERWORK/My Practical Work------------/Preprocessed Dataset recognition result/LNL model (dataset Dec)/LNL_26x1 evaluate with Experiment_21_12_2018_8Ch_-3,-2,0,5,7,16,17_1.5 bar_Leak_Test data/'
                        'C:/Users/YH/Desktop/hooyuheng.master/MASTER_PAPERWORK/My Practical Work------------/Preprocessed Dataset recognition result/LNL model (dataset Dec)/LNL_26x1 evaluate with Experiment_21_12_2018_8Ch_-3,-2,0,5,7,16,17_1.5 bar_NoLeak_Test data/'
                        'C:/Users/YH/Desktop/hooyuheng.master/MASTER_PAPERWORK/My Practical Work------------/Preprocessed Dataset recognition result/LNL model (dataset Dec)/LNL_26x1 evaluate with Experiment_21_12_2018_8Ch_-4,-2,2,4,6,8,10_1.5 bar_Leak_Test data/'
                        'C:/Users/YH/Desktop/hooyuheng.master/MASTER_PAPERWORK/My Practical Work------------/Preprocessed Dataset recognition result/LNL model (dataset Dec)/LNL_26x1 evaluate with Experiment_21_12_2018_8Ch_-4,-2,2,4,6,8,10_1.5 bar_NoLeak_Test data/'
                       )


declare -a noleak_label=(0 0 0 0 0 0 0)

declare -a leak_label=(1 1 1 1 1 1 1)

declare -a model_possible_input=(0 1)


echo ------------------------------------------------------------------------------------------- Training_2
python dataset_experiment_2019_1_3_\(LNL\)_train_script_2.py --model "${model_name}" --rfname "${result_save_filename}"

echo ------------------------------------------------------------------------------------------- Unseen leak_2
python dataset_experiment_2019_1_3_\(LNL\)_model_eval.py --model "${model_name}"\
                                                         --inlen ${in_length}\
                                                         --mpl ${model_possible_input[*]}\
                                                         --testdir "${test_tdms_dir[0]}"\
                                                         --dsf ${downsample_factor}\
                                                         --actlabel ${leak_label[*]}\
                                                         --inlabel ${unseen_data_labels[*]}\
                                                         --figname ${fig_label[0]}\
                                                         --rfname "${result_save_filename}"\
                                                         --savedircm "${cm_save_dir[0]}"\
                                                         --savedirpredcsv "${pred_result_save_dir[0]}"
#
echo ------------------------------------------------------------------------------------------- Unseen Noleak_2
python dataset_experiment_2019_1_3_\(LNL\)_model_eval.py --model "${model_name}"\
                                                         --inlen ${in_length}\
                                                         --mpl ${model_possible_input[*]}\
                                                         --testdir "${test_tdms_dir[1]}"\
                                                         --dsf ${downsample_factor}\
                                                         --actlabel ${noleak_label[*]}\
                                                         --inlabel ${unseen_data_labels[*]}\
                                                         --figname ${fig_label[1]}\
                                                         --rfname "${result_save_filename}"\
                                                         --savedircm "${cm_save_dir[1]}"\
                                                         --savedirpredcsv "${pred_result_save_dir[1]}"

echo ------------------------------------------------------------------------------------------- Seen leak_2
python dataset_experiment_2019_1_3_\(LNL\)_model_eval.py --model "${model_name}"\
                                                         --inlen ${in_length}\
                                                         --mpl ${model_possible_input[*]}\
                                                         --testdir "${test_tdms_dir[2]}"\
                                                         --dsf ${downsample_factor}\
                                                         --actlabel ${leak_label[*]}\
                                                         --inlabel ${seen_data_labels[*]}\
                                                         --figname ${fig_label[2]}\
                                                         --rfname "${result_save_filename}"\
                                                         --savedircm "${cm_save_dir[2]}"\
                                                         --savedirpredcsv "${pred_result_save_dir[2]}"

echo ------------------------------------------------------------------------------------------- Seen Noleak_2
python dataset_experiment_2019_1_3_\(LNL\)_model_eval.py --model "${model_name}"\
                                                         --inlen ${in_length}\
                                                         --mpl ${model_possible_input[*]}\
                                                         --testdir "${test_tdms_dir[3]}"\
                                                         --dsf ${downsample_factor}\
                                                         --actlabel ${noleak_label[*]}\
                                                         --inlabel ${seen_data_labels[*]}\
                                                         --figname ${fig_label[3]}\
                                                         --rfname "${result_save_filename}"\
                                                         --savedircm "${cm_save_dir[3]}"\
                                                         --savedirpredcsv "${pred_result_save_dir[3]}"

