# #!/bin/bash
# set -euo

# echo "==> Training..."


# #!/bin/sh 
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
#         . "/opt/conda/etc/profile.d/conda.sh"
#     else
#         export PATH="/opt/conda/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# conda activate slim
# cd /wanghuan/data/wangzefang/slim_VAR_copy/slimgpt_pub

# patch_nums_list=(1 2 3 4 5 6 8 10 13 16)
sample_list=(50 100 150 200)
result_txt="rebuttal1.log"
> $result_txt  # 清空输出文件

for num_samples in "${sample_list[@]}"; do
    
    # 1. 执行 prune.py
    specific_layer=256
    maxlayer=24 
    sparsity=0.2
    # num_samples=100
    model_name="d${maxlayer}_${sparsity}sparsity_${num_samples}i_${specific_layer}eva_scale.pth" 

    CUDA_VISIBLE_DEVICES=0 python -u /home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/prune.py \
    --minlayer 0 \
    --maxlayer $maxlayer \
    --num_samples $num_samples \
    --percdamp 1e-3 \
    --skip_evaluate \
    --prune_method slimgpt \
    --sparsity $sparsity \
    --specific_layer $specific_layer \
    --model_name $model_name

    # 2. 切换目录并执行 FID_test.py
    output_name=$model_name

    CUDA_VISIBLE_DEVICES=0 python -u /home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/FID_test.py --depth $maxlayer --sparsity $sparsity \
        --var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/${model_name}" \
        --output_name="${output_name}"

    # 3. 执行 torch-fidelity，并将输出追加到结果文件
    echo "num_samples: $num_samples" >> $result_txt

    echo "----- fidelity -----" >> $result_txt

    CUDA_VISIBLE_DEVICES=0 fidelity \
        --input1 "/home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/output/FID_test/${output_name}/" \
        --input2 "/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/virtual_images" \
        --fid \
        --kid \
        --isc \
        --gpu 0 >> $result_txt 2>&1

    echo "----- fidelity -----" >> $result_txt

    echo "----- pytorch_fid -----" >> $result_txt
    python -m pytorch_fid "/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/VIRTUAL_imagenet256_labeled.npz" "/home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/output/FID_test/${output_name}/" >> $result_txt 2>&1

    echo "----------------------------------------------" >> $result_txt

done


# fidelity --input1 "/wanghuan/data/wangzefang/VAR/FID_test/image/new_d24_0.2_1_input/"  --input2 "/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/virtual_images/" --fid --kid --isc --gpu 0

