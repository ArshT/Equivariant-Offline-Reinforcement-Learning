## Table of Contents
1. Installation
2. Data Collection
3. Training

We build this code base over the BulletArm benchmark.

<a name="install"></a>
## Install
1. Install dependencies
    ```
    pip install -r requirements.txt 
    pip install -r baseline_requirements.txt 
    ```
2. Add to your PYTHONPATH
    ```
    export PYTHONPATH=/path/to/Folder/:$PYTHONPATH
    ```
3. Test Installation
    ```
    python tutorials/block_stacking_demo.py
    ```



## Collecting Data
Navigate to the following directory: bulletarm_baselines/equi_rl/scripts/

1. Example of collecting a sub-optimal dataset for the block-in-bowl task with 10 demos with a reward limit of 0.4 (Medium)
```
python collect_demos_equi_sac_main.py  --algorithm equi_sacfd --max_train_step 200000 --env close_loop_block_in_bowl --data_collection  --data_collection_type sub_optimal --data_reward_limit 0.4  --data_demos 10 --num_eval_episodes 500 --eval_freq 1000 --seed 0 --batch_size 32 --device_name cuda:0  --num_objects 2
```

2. Example of collecting an optimal dataset for the block-in-bowl task with 10 expert demos
```
python collect_demos_equi_sac_main.py  --algorithm equi_sacfd --max_train_step 200000 --env close_loop_block_in_bowl --planner_episode 10 --data_collection  --data_collection_type optimal --data_expert_demos 10  --num_eval_episodes 500 --eval_freq 1000 --seed 0 --batch_size 32 --device_name cuda:0  --num_objects 2
```


## Training 
Navigate to the following directory: bulletarm_baselines/equi_rl/scripts/

1. Training Equi-CQL on a sub-optimal dataset for the block-in-bowl task with 10 demos with a reward limit of 0.4 (Medium)
```
python offline_main.py  --algorithm equi_cql --max_train_step 100000 --env close_loop_block_in_bowl  --data_collection_type sub_optimal --data_reward_limit 0.4  --data_demos 10 --num_eval_episodes 100 --eval_freq 1000 --seed 0 --batch_size 8 --device_name cuda:0  --num_objects 2
```


2. Training Non-Equi IQL on an optimal dataset for the block-in-bowl task with 10 expert demos
```
python offline_main.py  --algorithm iql --max_train_step 100000 --env close_loop_block_in_bowl --data_collection_type optimal --data_expert_demos 10  --num_eval_episodes 500 --eval_freq 1000 --seed 0 --batch_size 8 --device_name cuda:0  --num_objects 2 --iql_quantile 0.8
```
