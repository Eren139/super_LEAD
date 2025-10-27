array=(
    disc
    laiw
    lawbench
    self_built
)
for i in "${array[@]}"
do
    echo $i
        python /root/autodl-tmp/test/evaluation/generation/eva_generation.py \
            --dataset_name $i \
            --model_name_or_path /root/autodl-tmp/model/legal_quan_internlm \
            --max_length 1024

done