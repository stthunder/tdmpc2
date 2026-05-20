#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

python train.py \
	task=cheetah-run \
	source_task=mt30 \
	+pad_to_source_task=true \
	model_size=5 \
	world_model=mam_ode \
	model_only=true \
	model_epochs=500 \
	exp_name=mam_ode_cheetah_run \
	compile=false \
	model_history=30 \
	horizon=20 \
	state_model_coef=0.0 \
	reward_model_coef=1.0 \
	save_video=false
