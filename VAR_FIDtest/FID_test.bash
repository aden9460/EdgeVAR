depth=16
sparsity=0.2
num_samples=30
prune_method="slimgpt_20epoch"
output_name="d${depth}_${sparsity}sparsity_${num_samples}i_${prune_method}_method__temporary"
var_model="/home/wangzefang/.cache/huggingface/hub/models--liyy201912--d16_0.2_0-20_/snapshots/f0c9e04d64844445ad38357ac6bd960410506213/ar-ckpt-best.pth"
CUDA_VISIBLE_DEVICES=5 python FID_test.py --depth $depth --sparsity $sparsity --var_model=$var_model --output_name=$output_name

# var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d24_0.2var_${num_samples}i_256input_temporary.pth"