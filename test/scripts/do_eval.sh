array=(
    "D:\code\Superfiltering-main\test\scripts\logs\legal_ifd_internlm3883-VS-legal_quan_internlm\disc_wrap.json"
    "D:\code\Superfiltering-main\test\scripts\logs\legal_ifd_internlm3883-VS-legal_quan_internlm\laiw_wrap.json"
    "D:\code\Superfiltering-main\test\scripts\logs\legal_ifd_internlm3883-VS-legal_quan_internlm\lawbench_wrap.json"
    "D:\code\Superfiltering-main\test\scripts\logs\legal_ifd_internlm3883-VS-legal_quan_internlm\self_built_wrap.json"
)
for i in "${array[@]}"
do
    echo $i
    python "D:\\code\\Superfiltering-main\\test\\evaluation\\generation\\eval.py" \
        --wraped_file "$i" \
        --batch_size 10 \
        --api_key "sk-WKUQTNSUEWgdI7Vi2yEqzT6twUyqkr0yJl5WK5OVRyiqBZGi"
done
