import numpy as np
from PIL import Image
import os

# data = np.load('/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/VIRTUAL_imagenet256_labeled.npz')
# print(data['arr_0'].shape)
# print(data['arr_0'].dtype)
# arr = np.load('/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/VIRTUAL_imagenet256_labeled.npz')['images']
# save_dir = '/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/virtual_images'
# os.makedirs(save_dir, exist_ok=True)
# for i, img in enumerate(arr):
#     Image.fromarray(img).save(os.path.join(save_dir, f'{i:06d}.png'))


# 加载 npz 文件
data = np.load('/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/VIRTUAL_imagenet256_labeled.npz')
images = data['arr_0']

# 保存目录
save_dir = '/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/virtual_images/'
os.makedirs(save_dir, exist_ok=True)

# 循环保存图片
for i, img in enumerate(images):
    Image.fromarray(img).save(os.path.join(save_dir, f'{i:06d}.png'))

print(f"已保存 {len(images)} 张图片到 {save_dir}")