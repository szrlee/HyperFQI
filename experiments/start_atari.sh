cuda_id=$1
task=$2

alg_type=HyperFQI
for seed in 2025 2030 2035 2040
do
    sh experiments/atari/run_${alg_type}.sh ${cuda_id} ${seed} ${task}
    sleep 0.5
    echo "run $cuda_id $task"

    let cuda_id=$cuda_id+1
done

