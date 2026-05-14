#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

python train.py \
	task=walker-stand \
	source_task=mt30 \
	model_size=5 \
	world_model=mam_ode \
	model_only=true \
	model_epochs=500 \
	exp_name=mam_ode_walker_stand \
	compile=false \
	model_history=20 \
	horizon=20 \
	save_video=false
