cuda_id=$1
size=$2

alg_type=HyperFQI
for seed in 2020 2025
do
    sh experiments/deepsea/run_${alg_type}.sh $cuda_id 2020 $size
    sleep 0.5
    echo "run $cuda_id $task"

    let cuda_id=$cuda_id+1
done

