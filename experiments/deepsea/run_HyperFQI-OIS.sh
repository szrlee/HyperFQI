export CUDA_VISIBLE_DEVICES=$1
seed=$2

size=$3
if [ $size -eq 20 ]; then
    epoch=50
elif [ $size -eq 30 ]; then
    epoch=100
elif [ $size -eq 40 ]; then
    epoch=120
elif [ $size -eq 50 ]; then
    epoch=200
elif [ $size -eq 60 ]; then
    epoch=300
elif [ $size -eq 70 ]; then
    epoch=350
elif [ $size -eq 80 ]; then
    epoch=500
elif [ $size -eq 90 ]; then
    epoch=900
elif [ $size -eq 100 ]; then
    epoch=1000
elif [ $size -eq 120 ]; then
    epoch=1500
fi

task=DeepSea-v0
alg_type=HyperFQI-OIS
noise_dim=4
update_noise_per_sample=20
target_noise_per_sample=5

action_sample_num=5
action_select_scheme=MAX

for i in $(seq 5)
do
    tag=$(date "+%Y%m%d%H%M%S")
    python -m hyperfqi.scripts.run_deepsea --seed=${seed} --task=${task} --alg-type=${alg_type} \
    --noise-dim=${noise_dim} \
    --update-noise-per-sample=${update_noise_per_sample} \
    --target-noise-per-sample=${target_noise_per_sample} \
    --action-sample-num=${action_sample_num} \
    --action-select-scheme=${action_select_scheme} \
    --size=${size} \
    --epoch=${epoch} \
    > ~/logs/${alg_type}_${task}_${tag}.out 2> ~/logs/${alg_type}_${task}_${tag}.err &
    echo "run $seed $tag"
    let seed=$seed+1
    sleep 2
done

