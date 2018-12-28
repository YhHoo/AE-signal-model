
#!/bin/bash
declare -a batch_size=( 100 200 )
declare -a split=( 70 50 )

tdms_dir = 'F:/Experiment_21_12_2018/8Ch/-4,-2,2,4,6,8,10/2 bar/NoLeak/Train & Val data/'

for bs in ${batch_size[*]}
do
	for sp in ${split[*]}
	do
		echo -----------------------
		echo bs $bs sp $sp
		python test2.py --sen $bs --wrd $sp
	done
done