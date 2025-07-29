depth=16
sparsity=0.2
output_name="d${depth}_${sparsity}sparsity"
var_model="n"

CUDA_VISIBLE_DEVICES=0 python FID_test.py --depth $depth --sparsity $sparsity --var_model=$var_model --output_name=$output_name
