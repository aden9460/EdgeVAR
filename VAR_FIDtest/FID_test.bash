CUDA_VISIBLE_DEVICES=0 python FID_test.py --depth 24 --sparsity 0.2 --data_path="/datasets/liying/datasets/imagenet" \
 --var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d16_0.2_256_input_20i.pth" \
 --output_name="d16_0.2_uniform_20i_0epoch_rightprune" 
