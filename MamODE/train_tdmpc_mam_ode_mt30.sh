#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

python train.py \
	task=mt30 \
	model_size=5 \
	world_model=mam_ode \
	model_only=true \
	exp_name=mam_ode_mt30 \
	compile=false \
	save_video=false
