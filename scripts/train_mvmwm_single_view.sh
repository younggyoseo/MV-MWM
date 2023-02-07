TASK=$1
USE_ROTATION=$2
GPU=$3
SEED=$4

DISPLAY=:0.${GPU} TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python mvmwm/train.py \
--logdir ./logs/${TASK}/mvmwm_sv_rot${USE_ROTATION}/${SEED} \
--camera_keys 'front|wrist' \
--control_input front \
--task ${TASK} \
--mae.view_masking 1 \
--mae.viewpoint_pos_emb False \
--steps 302000 \
--num_demos 50 \
--use_rotation ${USE_ROTATION} \
--seed ${SEED}