#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../tdmpc2"

CHECKPOINT=${1:-logs/cheetah-run-front/1/mam_ode_cheetah_run_front_terminal80/models/epoch_240.pt}

if [[ ! -f "$CHECKPOINT" ]]; then
	echo "Checkpoint not found: $CHECKPOINT"
	echo "Current directory: $(pwd)"
	echo "Available cheetah-run-front MamODE checkpoints:"
	find logs/cheetah-run-front/1 -path '*/models/*.pt' 2>/dev/null | grep 'mam_ode_cheetah_run_front' | sort | tail -20 || true
	exit 1
fi

CHECKPOINT=$(realpath "$CHECKPOINT")
echo "Using checkpoint: $CHECKPOINT"

python evaluate_mam_ode.py \
	task=cheetah-run-front \
	source_task=mt30 \
	+pad_to_source_task=true \
	model_size=5 \
	world_model=mam_ode \
	checkpoint="$CHECKPOINT" \
	eval_episodes=1 \
	model_history=20 \
	horizon=20 \
	+mam_mpc_action_penalty=0.01 \
	+mam_mpc_delta_penalty=0.01 \
	mam_mpc_reward_weight=1.0 \
	+mam_mpc_terminal_weight=20.0 \
	+mam_mpc_print_plan=true \
	compile=false \
	save_video=true
