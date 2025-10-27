array=(
    disc
)
for i in "${array[@]}"
do
    echo $i
        python /root/autodl-tmp/test/evaluation/generation/eval_generation_wrap.py \
            --dataset_name $i \
            --fname1 /root/autodl-tmp/model/legal_filter_internlm5302/test_inference \
            --fname2 /root/autodl-tmp/model/legal_quan_internlm/test_inference \
            --save_name reward_dir5302-VS-quan \
            --max_length 1024
done

