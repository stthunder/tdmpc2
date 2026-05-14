#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

CHECKPOINT=${1:-logs/mt30/1/mam_ode_mt30/models/epoch_500.pt}

python evaluate_mam_ode.py \
	task=mt30 \
	model_size=5 \
	world_model=mam_ode \
	checkpoint="$CHECKPOINT" \
	eval_episodes=10 \
	model_history=20 \
	horizon=20 \
	compile=false \
	save_video=false
