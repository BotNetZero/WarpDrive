# WarpDrive
a LLM training engine under cluster, CS(Client-Server) environment

## tasks:
1. :building_construction: topology:
	- :white_check_mark: cluster mode
	- :building_construction: CS mode
2. :building_construction: distributed communication (optimize torch.distribtued.distributed_c10d)
	- :building_construction: group, subgroup
	- :white_check_mark: fix _store_based_barrier
	- :white_check_mark: P2P comm
3. :building_construction: pipeline parallel 
	- :white_check_mark: staged model
	- :building_construction: sequence pipeline schedule
4. :building_construction: activation recomputation
	- :building_construction: full mode
	- :stop_sign: selective mode
5. :stop_sign: data parallel
6. :stop_sign: tensor parallel
7. :stop_sign: sequence parallel
8. :building_construction: training data (open source)
	- :white_check_mark: OIG
	- :stop_sign: pile
9. :building_construction: models
	- :white_check_mark: Pythia7B
	- :stop_sign: parallel models
10. :building_construction: llm training
	- :building_construction: pretrain
	- :stop_sign: RLHF
	- :stop_sign: RLAI
11. :stop_sign: llm evaluation
12. :building_construction: mixed precision
	- :white_check_mark: empty model init, device map, partial loading
	- :white_check_mark: fp16
	- :stop_sign: int8


## GPUs topology
cluster环境下的均配结构: world_size = pp_size * dp_size * tp_size
![avatar](./docs/imgs/3D.jpg)

```xml
e.g.: 
world_size = 12
pp_size = 3
dp_size = 2
tp_size = 2
pp groups: [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
dp groups: [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]
tp groups: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
```


CS环境下的不均配结构: 按照各stage的GPUs组群
![avatar](./docs/imgs/hetero.jpg)
```xml
gpus = [1, 3, 3]
world_size = sum(gpus)
pp_size = len(gpus)
dp_size*tp_size = max(gpus)
ppg: [(0,1,4), (0,2,5), (0,3,6)]
dpg: [(1,2,3), (4,5,6)]
```

## Concept
1. process group
- main group: 区别pytorch.distributed的default pg, 可以有多个main group
- subgroup: main group可以有多个sub group, pp/dp/tp mode对应不同的subgroup

2. tasks
- forward computing
- backward computing
- send 
- receive
- scatter
- gather
- reduce

通信模式解释
![avatar](./docs/imgs/collective comm.jpg)


3. schedule
- pipeline schedule: 
	- sequence 
	- 1f1b w/o interleave
	- 1f1b with interleave

4. Mixed precision
- fp32
- fp16, bf16
- fp8 (nvidia transformer engine) with Hopper GPU architectur

## training data
Open source training data


## models
1. GPTNeoX
- Pythia7B


## todo
1. memory access
2. matrix swap
3. sparse transformer
