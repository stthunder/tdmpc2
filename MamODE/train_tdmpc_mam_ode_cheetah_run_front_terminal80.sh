#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

python train.py \
	task=cheetah-run-front \
	source_task=mt30 \
	+pad_to_source_task=true \
	model_size=5 \
	world_model=mam_ode \
	model_only=true \
	model_epochs=500 \
	exp_name=mam_ode_cheetah_run_front_terminal80_rewardctx \
	compile=false \
	model_history=20 \
	+mam_reward_context=true \
	horizon=20 \
	mam_terminal_horizon=20 \
	mam_terminal_value_coef=0 \
	state_model_coef=1.0 \
	reward_model_coef=1.0 \
	save_video=false
