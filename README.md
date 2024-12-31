**Paper**: [Equivariant Offline Reinforcement Learning](https://arxiv.org/abs/2406.13961)

**Abstract**: Sample efficiency is critical when applying learning-based methods to robotic manipulation due to the high cost of collecting expert demonstrations and the challenges of on-robot policy learning through online Reinforcement Learning (RL). Offline RL addresses this issue by enabling policy learning from an offline dataset collected using any behavioral policy, regardless of its quality. However, recent advancements in offline RL have predominantly focused on learning from large datasets. Given that many robotic manipulation tasks can be formulated as rotation-symmetric problems, we investigate the use of SO(2)-equivariant neural networks for offline RL with a limited number of demonstrations. Our experimental results show that equivariant versions of Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) outperform their non-equivariant counterparts. We provide empirical evidence demonstrating how equivariance improves offline learning algorithms in the low-data regime.

We build this repository based on the [BulletArm benchmark and BulletArm baselines](https://github.com/ColinKohler/BulletArm)

## Table of Contents
1. Installation
2. Data Collection
3. Training
 

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
python collect_demos_equi_sac_main.py  --algorithm equi_sacfd --max_train_step 200000 --env close_loop_block_in_bowl --data_collection_type sub_optimal --data_reward_limit 0.4  --data_demos 10 --num_eval_episodes 500 --eval_freq 1000 --seed 0 --batch_size 32 --device_name cuda:0  --num_objects 2
```

2. Example of collecting an optimal dataset for the block-in-bowl task with 10 expert demos
```
python collect_demos_equi_sac_main.py  --algorithm equi_sacfd --max_train_step 200000 --env close_loop_block_in_bowl --planner_episode 10 --data_collection_type optimal --data_expert_demos 10  --num_eval_episodes 500 --eval_freq 1000 --seed 0 --batch_size 32 --device_name cuda:0  --num_objects 2
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
