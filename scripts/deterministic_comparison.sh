echo "Hopper Distraction MBPO VAML"

let "SEED = $RANDOM"

echo "VAML distracted"
for i in {1..8};
do
	let "RUN_SEED = $SEED + $i"
	sbatch hopper_distraction/vaml_distr.sh $RUN_SEED 15 --error="$HOME/Claas/logs" --output="$HOME/Claas/logs"
done

echo "MLE distracted"
for i in {1..8};
do
	let "RUN_SEED = $SEED + $i"
	sbatch hopper_distraction/mle_distr.sh $RUN_SEED 15 --error="$HOME/Claas/logs" --output="$HOME/Claas/logs"
done

echo "MLE deterministic distracted"
for i in {1..8};
do
	let "RUN_SEED = $SEED + $i"
	sbatch hopper_distraction/mle_deterministic.sh $RUN_SEED 15 --error="$HOME/Claas/logs" --output="$HOME/Claas/logs"
done
