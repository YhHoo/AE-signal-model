#!/usr/bin/env bash

# -------------------------------------------------------------------------------------------------------- GLOBAL CONFIG
declare -a in_length=2000

declare -a downsample_factor=40

declare -a epoch=600

# ------------------------------------------------------------------------------------------------------ ITERABLE PARAM

declare -a model_name=(
                       'LD_9x4'
                       'LD_9x5'
                       'LD_9x6'
                       'LD_9x7'
                      )

declare -a rmsprop_rho=(0.8 0.6 0.4 0.2)

declare -a kernel_op1=(40 40 20 20 10 5)
declare -a kernel_op2=(40 40 20 20 5 5)
declare -a kernel_op3=(40 40 20 10 5 5)
declare -a fc_op1=(120 60)
declare -a fc_op2=(240 120)
declare -a fc_op3=(480 240)


echo start--------

for ((i=0;i<4;i++));
do
    echo ----------------------------------TRAINING WITH rho ${rmsprop_rho[i]} model "${model_name[i]}"
    python dataset_experiment_2019_1_3_\(LD\)_train_script.py --model "${model_name[i]}"\
                                                              --kernel_size ${kernel_op2[*]}\
                                                              --fc_size ${fc_op3[*]}\
                                                              --epoch ${epoch}\
                                                              --rmsprop_rho ${rmsprop_rho[i]}
done


