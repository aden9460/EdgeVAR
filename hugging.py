from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_file(
    path_or_fileobj="/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/model_zoo/vae_ch160v4096z32.pth",
    path_in_repo="vae.pth",
    repo_id="adendaaa/d16_0.2_0-20_",
    repo_type="model",
)