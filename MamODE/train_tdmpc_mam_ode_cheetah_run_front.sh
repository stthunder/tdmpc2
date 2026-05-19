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
	exp_name=mam_ode_cheetah_run_front \
	compile=false \
	model_history=20 \
	horizon=20 \
	save_video=false
