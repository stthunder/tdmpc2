#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

CHECKPOINT=${1:-logs/mt80/1/mam_ode_mt80/models/epoch_500.pt}

if [[ ! -f "$CHECKPOINT" ]]; then
	echo "Checkpoint not found: $CHECKPOINT"
	echo "Current directory: $(pwd)"
	echo "Available MamODE checkpoints:"
	find logs/mt80/1 -path '*/models/*.pt' 2>/dev/null | grep 'mam_ode_mt80' | sort | tail -20 || true
	exit 1
fi

CHECKPOINT=$(realpath "$CHECKPOINT")
echo "Using checkpoint: $CHECKPOINT"

python evaluate_mam_ode.py \
	task=mt80 \
	model_size=5 \
	world_model=mam_ode \
	checkpoint="$CHECKPOINT" \
	eval_episodes=1 \
	model_history=30 \
	horizon=20 \
	+mam_mpc_action_penalty=0.01 \
	+mam_mpc_delta_penalty=0.00 \
	mam_mpc_reward_weight=1.0 \
	+mam_mpc_terminal_weight=10.0 \
	+mam_mpc_print_plan=true \
	compile=false \
	save_video=true
