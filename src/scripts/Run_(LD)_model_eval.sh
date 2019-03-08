#!/usr/bin/env bash

# -------------------------------------------------------------------------------------------------------- GLOBAL CONFIG
declare -a in_length=2000

declare -a downsample_factor=40

declare -a epoch=600

# ------------------------------------------------------------------------------------------------------ ITERABLE PARAM
declare -a kernel_op1=(40 40 20 20 10 5)
declare -a kernel_op2=(40 40 20 20 5 5)
declare -a kernel_op3=(40 40 20 10 5 5)
declare -a fc_op1=(120 60)
declare -a fc_op2=(240 120)
declare -a fc_op3=(480 240)

# -------------------------------------------------------------------------------------------------------------- JOB 1
declare -a model_name='LD_9x3'  # **

echo ------------------------------------------------------------------------------------------- Training_JOB1
python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
                                                          --kernel_size ${kernel_op2[*]}\
                                                          --fc_size ${fc_op3[*]}\
                                                          --epoch ${epoch}

echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
                                                         --dsf ${downsample_factor}

## -------------------------------------------------------------------------------------------------------------- JOB 2
#declare -a model_name='LD_3x3'  # **
#
#echo ------------------------------------------------------------------------------------------- Training_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
#                                                          --kernel_size ${kernel_op1[*]}\
#                                                          --fc_size ${fc_op2[*]}\
#                                                          --epoch ${epoch}
#
#echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
#                                                         --dsf ${downsample_factor}
#
## -------------------------------------------------------------------------------------------------------------- JOB 3
#declare -a model_name='LD_3x4'  # **
#
#echo ------------------------------------------------------------------------------------------- Training_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
#                                                          --kernel_size ${kernel_op1[*]}\
#                                                          --fc_size ${fc_op3[*]}\
#                                                          --epoch ${epoch}
#
#echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
#                                                         --dsf ${downsample_factor}
#
## -------------------------------------------------------------------------------------------------------------- JOB 4
#declare -a model_name='LD_3x5'  # **
#
#echo ------------------------------------------------------------------------------------------- Training_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
#                                                          --kernel_size ${kernel_op2[*]}\
#                                                          --fc_size ${fc_op1[*]}\
#                                                          --epoch ${epoch}
#
#echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
#                                                         --dsf ${downsample_factor}
#
## -------------------------------------------------------------------------------------------------------------- JOB 5
#declare -a model_name='LD_3x6'  # **
#
#echo ------------------------------------------------------------------------------------------- Training_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
#                                                          --kernel_size ${kernel_op2[*]}\
#                                                          --fc_size ${fc_op2[*]}\
#                                                          --epoch ${epoch}
#
#echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
#                                                         --dsf ${downsample_factor}
#
## -------------------------------------------------------------------------------------------------------------- JOB 6
#declare -a model_name='LD_3x7'  # **
#
#echo ------------------------------------------------------------------------------------------- Training_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
#                                                          --kernel_size ${kernel_op2[*]}\
#                                                          --fc_size ${fc_op3[*]}\
#                                                          --epoch ${epoch}
#
#echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
#                                                         --dsf ${downsample_factor}
#
## -------------------------------------------------------------------------------------------------------------- JOB 7
#declare -a model_name='LD_3x8'  # **
#
#echo ------------------------------------------------------------------------------------------- Training_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
#                                                          --kernel_size ${kernel_op3[*]}\
#                                                          --fc_size ${fc_op1[*]}\
#                                                          --epoch ${epoch}
#
#echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
#                                                         --dsf ${downsample_factor}
#
## -------------------------------------------------------------------------------------------------------------- JOB 8
#declare -a model_name='LD_3x9'  # **
#
#echo ------------------------------------------------------------------------------------------- Training_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
#                                                          --kernel_size ${kernel_op3[*]}\
#                                                          --fc_size ${fc_op2[*]}\
#                                                          --epoch ${epoch}
#
#echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
#                                                         --dsf ${downsample_factor}
#
#
## -------------------------------------------------------------------------------------------------------------- JOB 10
#declare -a model_name='LD_3x10'  # **
#
#echo ------------------------------------------------------------------------------------------- Training_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
#                                                          --kernel_size ${kernel_op3[*]}\
#                                                          --fc_size ${fc_op3[*]}\
#                                                          --epoch ${epoch}
#
#echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
#python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
#                                                         --dsf ${downsample_factor}