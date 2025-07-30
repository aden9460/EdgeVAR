from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_file(
    path_or_fileobj="/home/wangzefang/edgevar/EdgeVAR/VAR/traind_model/d16_0.2_0-20_200i_temporary/ar-ckpt-last.pth",
    path_in_repo="ar-ckpt-last.pth",
    repo_id="adendaaa/d16_0.2_0-20_",
    repo_type="model",
)