from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='/home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/output/FID_test/d16_0.2_uniform_20i_0epoch_rightprune',
    input2='/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/virtual_images',
    cuda=True,  # 如果要用GPU，改为True
    fid=True,
    kid=True,
    verbose=True,
    isc=True
)

print(metrics)
