specific_layer=256
model_name="d24_0.4real_${specific_layer}_input_5i.pth"

CUDA_VISIBLE_DEVICES=0 python -u /wanghuan/data/wangzefang/slim_VAR_copy/slimgpt_pub/prune.py \
  --minlayer 0 \
  --maxlayer 24 \
  --num_samples 50 \
  --seqlen 256 \
  --percdamp 1e-3 \
  --min_sparsity 0.0625 \
  --max_sparsity 0.3 \
  --non_uniform \
  --non_uniform_strategy linear_decrease \
  --skip_evaluate \
  --prune_method slimgpt \
  --sparsity 0.4 \
  --specific_layer $specific_layer \
  --model_name $model_name
