
#!/bin/bash
declare -a batch_size=( 100 200 )
declare -a split=( 70 50 )


for bs in ${batch_size[*]}
do
	for sp in ${split[*]}
	do
		echo -----------------------
		echo bs $bs sp $sp
		python test2.py --sen $bs --wrd $sp
	done
done