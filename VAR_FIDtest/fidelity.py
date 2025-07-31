from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input2='/home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/output/FID_test/d16_0.2sparsity_150i_256eva_scale_taylor_temporary.pth',
    input1='/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/virtual_images',
    cuda=True,  # 如果要用GPU，改为True
    fid=True,
    isc=True,
    prc=True
)

print(metrics)
