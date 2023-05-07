# WarpDrive
a LLM training engine

## tasks:
1. distributed communication :white_check_mark:
2. pipeline parallel
3. data parallel
4. tensor parallel
5. sequence parallel

## concept
1. process group
- main group: 区别pytorch.distributed的default pg, 可以有多个main group
- subgroup: main group可以有多个sub group, pp/dp/tensor mode对应不同的subgroup

2. task
- forward computing
- backward computing
- send 
- receive

3. schedule
- pipeline schedule: sequence, 1f1b, ...

