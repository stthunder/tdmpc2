#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

CHECKPOINT=${1:-logs/mt30/1/mam_ode_mt30/models/epoch_300.pt}

if [[ ! -f "$CHECKPOINT" ]]; then
	echo "Checkpoint not found: $CHECKPOINT"
	echo "Current directory: $(pwd)"
	echo "Available MamODE checkpoints:"
	find logs/mt30/1 -path '*/models/*.pt' | grep 'mam_ode_mt30' | sort | tail -20
	exit 1
fi

CHECKPOINT=$(realpath "$CHECKPOINT")
echo "Using checkpoint: $CHECKPOINT"

python evaluate_mam_ode.py \
	task=mt30 \
	model_size=5 \
	world_model=mam_ode \
	checkpoint="$CHECKPOINT" \
	eval_episodes=1 \
	model_history=20 \
	horizon=20 \
	+mam_mpc_action_penalty=0.001 \
	+mam_mpc_delta_penalty=0.01 \
	+mam_mpc_terminal_weight=1.0 \
	compile=false \
	save_video=true
