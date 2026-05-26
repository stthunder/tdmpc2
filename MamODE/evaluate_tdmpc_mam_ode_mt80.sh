#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

CHECKPOINT=${1:-logs/mt80/1/mam_ode_mt80_terminal80/models/epoch_500.pt}

python evaluate_mam_ode.py \
	task=mt80 \
	model_size=5 \
	world_model=mam_ode \
	checkpoint="$CHECKPOINT" \
	eval_episodes=10 \
	model_history=20 \
	horizon=20 \
	mam_mpc_reward_weight=1.0 \
	mam_mpc_learned_terminal_weight=1.0 \
	compile=false \
	save_video=false
