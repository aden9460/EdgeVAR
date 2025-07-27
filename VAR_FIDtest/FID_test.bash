CUDA_VISIBLE_DEVICES=0 python FID_test.py --depth 24 --sparsity 0.2 --data_path="/datasets/liying/datasets/imagenet" \
 --var_model="/wanghuan/data/wangzefang/slim_VAR_copy/VAR/d20_0.2_0-20/ar-ckpt-last.pth" \
 --output_name="new_d24_0.2_uniform_0-20epoch" 
