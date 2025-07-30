depth=24
sparsity=0.2
num_samples=30
prune_method="taylor"
output_name="d${depth}_${sparsity}sparsity_${num_samples}i_${prune_method}_method_nocompensate_temporary"
var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d24_0.2sparsity_10i_256eva_scale_taylormethod_temporary.pth"
CUDA_VISIBLE_DEVICES=1 python FID_test.py --depth $depth --sparsity $sparsity --var_model=$var_model --output_name=$output_name

# var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d24_0.2var_${num_samples}i_256input_temporary.pth"