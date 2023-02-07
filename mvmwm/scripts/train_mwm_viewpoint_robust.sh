TASK=$1
USE_ROTATION=$2
DIFFICULTY=$3
GPU=$4
SEED=$5

DISPLAY=:0.${GPU} TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python mvmwm/train.py \
--logdir ./logs/${TASK}/mwm_${DIFFICULTY}/${SEED} \
--configs ${DIFFICULTY} eval_strong \
--use_randomize True \
--default_texture canonical \
--camera_keys 'cam1' \
--control_input cam1 \
--task ${TASK} \
--mae.view_masking 0 \
--mae.viewpoint_pos_emb False \
--steps 502000 \
--num_demos 100 \
--use_rotation ${USE_ROTATION} \
--augment True \
--seed ${SEED}