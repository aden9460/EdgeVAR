depth=16
sparsity=0.4
num_samples=30
prune_method="nocompensate"
output_name="d${depth}_${sparsity}sparsity_${num_samples}i_${prune_method}_method__temporary"
var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d16_0.4sparsity_150i_256eva_scale_slimgptnocompensate_temporary.pth"
CUDA_VISIBLE_DEVICES=2 python FID_test.py --depth $depth --sparsity $sparsity --var_model=$var_model --output_name=$output_name

# var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d24_0.2var_${num_samples}i_256input_temporary.pth"