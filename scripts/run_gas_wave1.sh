#!/bin/bash

WAVE_START=$(date +%s)

echo "=========================================="
echo "GAS-DRO WAVE 1 START"
echo "START_TIME=$(date -Is)"
echo "4 runs / 2 GPUs / 2 runs per GPU"
echo "=========================================="

run_one () {
    GPU=$1
    SEED=$2
    BUDGET=$3
    TAG=$4
    LOG=$5

    START=$(date +%s)

    {
        echo "=========================================="
        echo "START_TIME=$(date -Is)"
        echo "GPU=$GPU"
        echo "SEED=$SEED"
        echo "BUDGET=$BUDGET"
        echo "RUN_TAG=$TAG"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
        python experiments/run_gas_dro_tunable.py \
            --seed $SEED \
            --train-csv data/nominal/train.csv \
            --val-root data/ood/validation \
            --run-tag $TAG \
            --budget $BUDGET \
            --device cuda

        STATUS=$?

        END=$(date +%s)
        ELAPSED=$((END-START))
        H=$((ELAPSED/3600))
        M=$(((ELAPSED%3600)/60))
        S=$((ELAPSED%60))

        echo "=========================================="
        echo "END_TIME=$(date -Is)"
        echo "EXIT_CODE=$STATUS"
        echo "ELAPSED_SECONDS=$ELAPSED"
        printf "ELAPSED=%02d:%02d:%02d\n" $H $M $S
        echo "=========================================="

        exit $STATUS
    } > "$LOG" 2>&1
}

# GPU 0 - 두 개
run_one 0 0 0.010 budget_0p010 logs/gas_budget001_seed0.log &
PID0=$!

run_one 0 2 0.010 budget_0p010 logs/gas_budget001_seed2.log &
PID2=$!

# GPU 1 - 두 개
run_one 1 1 0.010 budget_0p010 logs/gas_budget001_seed1.log &
PID1=$!

run_one 1 3 0.010 budget_0p010 logs/gas_budget001_seed3.log &
PID3=$!

echo "PID seed0: $PID0"
echo "PID seed1: $PID1"
echo "PID seed2: $PID2"
echo "PID seed3: $PID3"

wait $PID0
S0=$?

wait $PID1
S1=$?

wait $PID2
S2=$?

wait $PID3
S3=$?

WAVE_END=$(date +%s)
WAVE_ELAPSED=$((WAVE_END-WAVE_START))

H=$((WAVE_ELAPSED/3600))
M=$(((WAVE_ELAPSED%3600)/60))
S=$((WAVE_ELAPSED%60))

{
    echo "=========================================="
    echo "GAS-DRO WAVE 1 COMPLETE"
    echo "END_TIME=$(date -Is)"
    echo "seed0 exit=$S0"
    echo "seed1 exit=$S1"
    echo "seed2 exit=$S2"
    echo "seed3 exit=$S3"
    echo "WAVE_ELAPSED_SECONDS=$WAVE_ELAPSED"
    printf "WAVE_ELAPSED=%02d:%02d:%02d\n" $H $M $S
    echo "=========================================="
} | tee logs/gas_wave1_summary.log

