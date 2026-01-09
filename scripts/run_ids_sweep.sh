#!/usr/bin/env bash
set -e

mkdir -p results/ids/summary

# Ρυθμίσεις sweep (άλλαξέ τα αν θες)
T=200
TSTART=100
SEEDS=$(seq 0 29)

# Attack modes που έχουμε
MODES=("none" "replay" "step_bias" "ramp_bias" "burst_noise")

# IDS params (παράδειγμα)
ALPHAS=(0.90 0.95 0.99)
NS=(1 2 3)

for mode in "${MODES[@]}"; do
  for a in "${ALPHAS[@]}"; do
    for n in "${NS[@]}"; do
      for s in $SEEDS; do

        # mode-specific args
        extra=""
        if [[ "$mode" == "replay" ]]; then
          extra="--replay_len 30"
        elif [[ "$mode" == "step_bias" ]]; then
          extra="--bias 0.3,0.3,0.15"
        elif [[ "$mode" == "ramp_bias" ]]; then
          extra="--rate 0.01,0.01,0.005"
        elif [[ "$mode" == "burst_noise" ]]; then
          extra="--sigma 0.4 --duration 20"
        fi

        echo "RUN seed=$s mode=$mode alpha=$a N=$n"
        python3 eval_ids_replay.py \
          --seed "$s" --T "$T" --t_start "$TSTART" \
          --mode "$mode" $extra \
          --alpha "$a" --n "$n"

      done
    done
  done
done

# μετά το sweep, μάζεψε τα αποτελέσματα
python3 collect_ids_summary.py
