#!/bin/bash

args=$1;
echo "arguments:"${args}

python3 -m src.main.llm_pretrain ${args}