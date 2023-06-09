#!/bin/bash

host_ip=$1;
host_port=$2;
tcp_url="tcp://${host_ip}:${host_port}"
echo "tcp url: ${tcp_url}"

stage=$3;
echo "pp stage: ${stage}"

local_rank=$4;
echo "local rank: ${local_rank}"

cuda_id=$5;
echo "cuda index: ${cuda_id}"

datasets="\
OIG/unified_chip2.jsonl:0.1,\
OIG/unified_conv_finqa.jsonl:0.1 \
"

export SHOW_DATA=0


args="--group_name usr \
--mode cluster \
--pp_backend nccl \
--dp_backend nccl \
--tp_backend nccl \
--dist_url ${tcp_url} \
--task_name ${datasets} \
--gpus 1 1 \
--first_last_stage_merge False \
--stage ${stage} \
--data_group_size 1 \
--tensor_group_size 1 \
--local_rank ${local_rank} \
--cuda_id ${cuda_id}
--recompute_activations \
--num_layers 10 \
--batch_size 64 \
--micro_batch_num 8 \
--recompute_activations \
--recompute_granularity full \
--warmup_steps 10 \
--total_steps 20000 \
--optimizer adamw \

"

echo "args: "${args}

echo "Pretrain Pythia7B"
docker run -it --rm -p 4001:4001 --network host -v /home/docker/warpdrive:/workspace -v /home/docker/models/pythia_7b:/workspace/saved_models/pythia_7b --runtime=nvidia --gpus all --name warpdrive diniu/assistant_base:v1 bash scripts/run_warpdrive.sh "${args}"



echo "over...."

# docker run -it --rm -p 4001:4001 --network host -v /home/docker/warpdrive:/workspace -v /home/docker/models/pythia_7b:/workspace/saved_models/pythia_7b --runtime=nvidia --gpus all --name assistant diniu/assistant_base:v1 bash
# python3 -m bot --model /workspace/models/Pythia-Chat-Base-7B -g 0:10 -r 10
