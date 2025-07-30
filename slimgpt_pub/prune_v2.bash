specific_layer=256
maxlayer=24
sparsity=0.2
num_samples=10
prune_method="taylor"
model_name="d${maxlayer}_${sparsity}sparsity_${num_samples}i_${specific_layer}eva_scale_${prune_method}method_temporary.pth" 

CUDA_VISIBLE_DEVICES=1 python -u /home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/prune_v2.py \
  --minlayer 0 \
  --maxlayer $maxlayer \
  --num_samples $num_samples \
  --percdamp 1e-3 \
  --skip_evaluate \
  --prune_method $prune_method \
  --sparsity $sparsity \
  --specific_layer $specific_layer \
  --model_name $model_name \
  # --seqlen $seqlen \
  # --min_sparsity 0.0625 \
  # --max_sparsity 0.3 \
  # --non_uniform \
  # --non_uniform_strategy linear_decrease \