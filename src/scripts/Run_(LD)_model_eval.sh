#!/usr/bin/env bash

# -------------------------------------------------------------------------------------------------------- GLOBAL CONFIG
declare -a in_length=2000

declare -a downsample_factor=40

# ------------------------------------------------------------------------------------------------------ ITERABLE PARAM
declare -a kernel_op1=(40 40 20 20 10 10)
declare -a kernel_op2=(5 5 5 5)
declare -a kernel_op3=(7 7 7 7)
declare -a fc_op1=(120 60)
declare -a fc_op2=(240 120)
declare -a fc_op3=(480 240)

# -------------------------------------------------------------------------------------------------------------- JOB 1
declare -a model_name='LD_3x1'  # **

echo ------------------------------------------------------------------------------------------- Training_JOB1
python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name}"\
                                                          --kernel_size ${kernel_op1[*]}\
                                                          --fc_size ${fc_op2[*]}

echo ------------------------------------------------------------------------------------------- Evaluate_JOB1
python dataset_experiment_2019_1_3_\(LD\)_model_eval.py --model "${model_name}"\
                                                         --dsf ${downsample_factor}

