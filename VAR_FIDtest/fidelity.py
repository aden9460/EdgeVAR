from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='/wanghuan/data/wangzefang/VAR/FID_test/image/new_d24_0.4_14epoch/',
    input2='/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/virtual_images/',
    cuda=True,  # 如果要用GPU，改为True
    fid=True,
    kid=True,
    verbose=True,
    isc=True
)

print(metrics)
