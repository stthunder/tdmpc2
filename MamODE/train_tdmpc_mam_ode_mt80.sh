#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

python train.py \
	task=mt80 \
	source_task=mt80 \
	data_dir=/home/lizhaoyang/Desktop/tdmpc2/datasets/mt80 \
	model_size=5 \
	world_model=mam_ode \
	model_only=true \
	model_epochs=500 \
	exp_name=mam_ode_mt80 \
	compile=false \
	model_history=30 \
	horizon=20 \
	state_model_coef=1.0 \
	reward_model_coef=0.1 \
	save_video=false
