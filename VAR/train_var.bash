# echo "==> Training..."


# #!/bin/sh 
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
#         . "/opt/conda/etc/profile.d/conda.sh"
#     else
#         export PATH="/opt/conda/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# conda activate slim

# # 训练参数
# TRAIN_SCRIPT="train.py"
# TRAIN_ARGS="--depth=24 --bs=240 --ep=20 --fp16=1 --sparsity=0.2 --local_out_dir_path="/wanghuan/data/wangzefang/slim_VAR_copy/VAR/d20_0.2_0-20" --data_path="/wanghuan/data/wangzefang/ImageNet-1K/""

# # 配置弹性参数
# NNODES="1:4"
# NPROC_PER_NODE=6
# MAX_RESTARTS=100

# # 用平台自动注入的job id作为rdzv_id，确保所有节点一样
# RDZV_ID=${VC_JOB_ID:-myjob20240513}
# RDZV_BACKEND="c10d"
# RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"

# # 打印环境变量，方便排查
# echo "MASTER_ADDR=$MASTER_ADDR"
# echo "MASTER_PORT=$MASTER_PORT"
# echo "RDZV_ID=$RDZV_ID"

# torchrun \
#   --nnodes=$NNODES \
#   --nproc_per_node=$NPROC_PER_NODE \
#   --max_restarts=$MAX_RESTARTS \
#   --rdzv_id=$RDZV_ID \
#   --rdzv_backend=$RDZV_BACKEND \
#   --rdzv_endpoint=$RDZV_ENDPOINT \
#   $TRAIN_SCRIPT $TRAIN_ARGS





CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun  \
  --nnodes=1 \
  --nproc_per_node=6 \
  --node_rank=0 \
  train.py \
  --depth=16 --bs=370 --ep=20 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.2 --local_out_dir_path="/home/wangzefang/edgevar/EdgeVAR/VAR/traind_model/d16_0.2_0-20_200i_temporary" --data_path="/home/wangzefang/Datasets/ImageNet-1K" \
  --var_path=""

# torchrun \
#   --nproc_per_node=6 \
#   --nnodes=2 \
#   --node_rank=1 \
#  train.py \
#   --depth=24 --bs=252 --ep=10 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.2 --local_out_dir_path="/wanghuan/data/wangzefang/slim_VAR_copy/VAR/d20_0.2_0-10" --data_path="/wanghuan/data/wangzefang/ImageNet-1K/"
