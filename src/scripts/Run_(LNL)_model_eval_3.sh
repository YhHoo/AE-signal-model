#!/usr/bin/env bash
# ITERATIVE TASK

# -------------------------------------------------------------------------------------------------------- GLOBAL CONFIG
declare -a in_length=2000

declare -a downsample_factor=40

declare -a epoch=200

# ------------------------------------------------------------------------------------------------------ ITERABLE PARAM

declare -a model_name=(
                       'LD_44x6'
                      )

declare -a rmsprop_rho=(0.8 0.6 0.4 0.2)
declare -a l2_layer=(0 1 2 3 4)

declare -a kernel_op1=(40 40 20 20 10 5)
declare -a kernel_op2=(40 40 20 20 5 5)
declare -a kernel_op3=(40 40 20 10 5 5)
declare -a fc_op1=(120 60)
declare -a fc_op2=(240 120)
declare -a fc_op3=(480 240)


echo start--------

for ((i=0;i<5;i++));
do
#    echo ----------------------------------TRAINING model "${model_name[i]}"
#    python dataset_experiment_2019_1_3_\(LNL\)_train_script.py --model "${model_name[i]}"\
#                                                               --kernel_size ${kernel_op2[*]}\
#                                                               --fc_size ${fc_op3[*]}\
#                                                               --epoch ${epoch}\

    echo ----------------------------------EVALUATING model "${model_name[i]}"
    python dataset_experiment_2019_1_3_\(LNL\)_model_eval_3.py --model "${model_name[i]}"\
                                                               --dsf ${downsample_factor}
done


