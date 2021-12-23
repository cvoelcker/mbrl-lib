echo "Ant ICLR rebuttal"

let "SEED = $RANDOM"

# echo "VAML"
# for i in {1..6};
# do
# 	let "RUN_SEED = $SEED + $i"
# 	sbatch ant/vaml_full.sh $RUN_SEED 
# done

echo "MLE"
for i in {1..6};
do
	let "RUN_SEED = $SEED + $i"
	sbatch ant/mle_full.sh $RUN_SEED 
done

