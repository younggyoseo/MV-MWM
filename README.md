# Multi-View Masked World Models for Visual Robotic Manipulation

Implementation of [MV-MWM](https://arxiv.org/abs/2302.02408) in TensorFlow 2.

## Method
Multi-View Masked World Models (MV-MWM) is a reinforcement learning framework that (i) trains a multi-view masked autoencoder with view-masking and (ii) learns a world model for single-view, multi-view, and viewpoint-robust control.

![MV-MWM Overview](https://user-images.githubusercontent.com/20944657/217286929-23c4bf7b-17e0-498a-b4b0-ace8d08fe118.gif)

## Instructions

Install dependencies
```
source dependency.sh
```

First install dependencies from [RLBench](https://github.com/stepjam/RLBench) repository. Then, install our customized RLBench in `rlbench_shaped_rewards` directory. 

```
cd ./rlbench_shaped_rewards
pip install -e .
```

## Experiments

To reproduce our experiments, please run below scripts in `mvmwm` directory.

### Multi-View Control
```
source ./scripts/train_mvmwm_multi_view.sh {TASK} {USE_ROTATION} {GPU} {SEED}
# For instance,
source ./scripts/train_mvmwm_multi_view.sh rlbench_phone_on_base False 0 1
source ./scripts/train_mvmwm_multi_view.sh rlbench_stack_wine True 0 1
```

### Single-View Control
```
source ./scripts/train_mvmwm_single_view.sh {TASK} {USE_ROTATION} {GPU} {SEED}
# For instance,
source ./scripts/train_mvmwm_single_view.sh rlbench_phone_on_base False 0 1
source ./scripts/train_mvmwm_single_view.sh rlbench_stack_wine True 0 1
```

### Viewpoint-Robust Control
```
source ./scripts/train_mvmwm_viewpoint_robust.sh {TASK} {USE_ROTATION} {DIFFICULTY} {GPU} {SEED}
# For instance,
source ./scripts/train_mvmwm_viewpoint_robust.sh rlbench_phone_on_base_custom False medium 0 1
source ./scripts/train_mvmwm_viewpoint_robust.sh rlbench_stack_wine_custom True weak 0 1
```
