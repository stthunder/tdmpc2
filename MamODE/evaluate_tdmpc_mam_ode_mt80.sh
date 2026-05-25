#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

CHECKPOINT=${1:-logs/mt80/1/mam_ode_mt80/models/epoch_500.pt}

python evaluate_mam_ode.py \
	task=mt80 \
	model_size=5 \
	world_model=mam_ode \
	mam_ode_solver=torchdiffeq \
	mam_ode_method=rk4 \
	checkpoint="$CHECKPOINT" \
	eval_episodes=10 \
	model_history=30 \
	horizon=6 \
	mam_mpc_time_points=1_2_3_10_40_80 \
	mam_mpc_reward_weight=1.0 \
	compile=false \
	save_video=false
