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

script=$6
echo "test script: ${script}"


args="--group_name usr \
--mode cluster \
--pp_backend nccl \
--dp_backend nccl \
--tp_backend nccl \
--dist_url ${tcp_url} \
--gpus 1 1 \
--first_last_stage_merge False \
--stage ${stage} \
--data_group_size 1 \
--tensor_group_size 1 \
--local_rank ${local_rank} \
--cuda_id ${cuda_id}
"

echo "args: "${args}

docker run -it --rm -p 4001:4001 -v /home/docker/warpdrive:/workspace --runtime=nvidia --gpus all --name warpdrive diniu/assistant_base:v1 bash python3 -m src.test.${script} "${args}"



echo "over...."

# docker run -it --rm -p 4001:4001 -v /home/docker/warpdrive:/workspace -v /home/docker/models/Pythia-Chat-Base-7B:/workspace/models/Pythia-Chat-Base-7B --runtime=nvidia --gpus all --name assistant diniu/assistant_base:v1 bash
# python3 -m bot --model /workspace/models/Pythia-Chat-Base-7B -g 0:10 -r 10