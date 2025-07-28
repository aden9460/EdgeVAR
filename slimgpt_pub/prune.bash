specific_layer=256
model_name="d16_0.2_${specific_layer}_input_100i.pth"

CUDA_VISIBLE_DEVICES=2 python -u /home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/prune.py \
  --minlayer 0 \
  --maxlayer 16 \
  --num_samples 100 \
  --seqlen 256 \
  --percdamp 1e-3 \
  --min_sparsity 0.0625 \
  --max_sparsity 0.3 \
  --skip_evaluate \
  --prune_method slimgpt \
  --sparsity 0.2 \
  --specific_layer $specific_layer \
  --model_name $model_name

  # --non_uniform \
  # --non_uniform_strategy linear_decrease \