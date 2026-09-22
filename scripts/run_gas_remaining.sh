#!/bin/bash

TOTAL_START=$(date +%s)

echo "=========================================="
echo "GAS-DRO REMAINING 11 RUNS START"
echo "START_TIME=$(date -Is)"
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


# ============================================================
# WAVE 2
#   GPU 0 : budget .010 seed4 / budget .015 seed0
#   GPU 1 : budget .015 seed1 / budget .015 seed2
# ============================================================

echo
echo "=========================================="
echo "WAVE 2 START : $(date -Is)"
echo "=========================================="

WAVE_START=$(date +%s)

run_one 0 4 0.010 budget_0p010 logs/gas_budget001_seed4.log &
P1=$!

run_one 0 0 0.015 budget_0p015 logs/gas_budget0015_seed0.log &
P2=$!

run_one 1 1 0.015 budget_0p015 logs/gas_budget0015_seed1.log &
P3=$!

run_one 1 2 0.015 budget_0p015 logs/gas_budget0015_seed2.log &
P4=$!

wait $P1; S1=$?
wait $P2; S2=$?
wait $P3; S3=$?
wait $P4; S4=$?

WAVE_END=$(date +%s)
WAVE_ELAPSED=$((WAVE_END-WAVE_START))

printf "WAVE2_ELAPSED=%02d:%02d:%02d\n" \
    $((WAVE_ELAPSED/3600)) \
    $(((WAVE_ELAPSED%3600)/60)) \
    $((WAVE_ELAPSED%60))

echo "WAVE2_EXIT_CODES=$S1,$S2,$S3,$S4"


# ============================================================
# WAVE 3
#   GPU 0 : budget .015 seed3 / budget .015 seed4
#   GPU 1 : budget .020 seed0 / budget .020 seed1
# ============================================================

echo
echo "=========================================="
echo "WAVE 3 START : $(date -Is)"
echo "=========================================="

WAVE_START=$(date +%s)

run_one 0 3 0.015 budget_0p015 logs/gas_budget0015_seed3.log &
P1=$!

run_one 0 4 0.015 budget_0p015 logs/gas_budget0015_seed4.log &
P2=$!

run_one 1 0 0.020 budget_0p020 logs/gas_budget002_seed0.log &
P3=$!

run_one 1 1 0.020 budget_0p020 logs/gas_budget002_seed1.log &
P4=$!

wait $P1; S1=$?
wait $P2; S2=$?
wait $P3; S3=$?
wait $P4; S4=$?

WAVE_END=$(date +%s)
WAVE_ELAPSED=$((WAVE_END-WAVE_START))

printf "WAVE3_ELAPSED=%02d:%02d:%02d\n" \
    $((WAVE_ELAPSED/3600)) \
    $(((WAVE_ELAPSED%3600)/60)) \
    $((WAVE_ELAPSED%60))

echo "WAVE3_EXIT_CODES=$S1,$S2,$S3,$S4"


# ============================================================
# WAVE 4
#   GPU 0 : budget .020 seed2 / budget .020 seed3
#   GPU 1 : budget .020 seed4
# ============================================================

echo
echo "=========================================="
echo "WAVE 4 START : $(date -Is)"
echo "=========================================="

WAVE_START=$(date +%s)

run_one 0 2 0.020 budget_0p020 logs/gas_budget002_seed2.log &
P1=$!

run_one 0 3 0.020 budget_0p020 logs/gas_budget002_seed3.log &
P2=$!

run_one 1 4 0.020 budget_0p020 logs/gas_budget002_seed4.log &
P3=$!

wait $P1; S1=$?
wait $P2; S2=$?
wait $P3; S3=$?

WAVE_END=$(date +%s)
WAVE_ELAPSED=$((WAVE_END-WAVE_START))

printf "WAVE4_ELAPSED=%02d:%02d:%02d\n" \
    $((WAVE_ELAPSED/3600)) \
    $(((WAVE_ELAPSED%3600)/60)) \
    $((WAVE_ELAPSED%60))

echo "WAVE4_EXIT_CODES=$S1,$S2,$S3"


# ============================================================
# TOTAL
# ============================================================

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END-TOTAL_START))

echo
echo "=========================================="
echo "ALL REMAINING GAS-DRO RUNS COMPLETE"
echo "END_TIME=$(date -Is)"
printf "TOTAL_ELAPSED=%02d:%02d:%02d\n" \
    $((TOTAL_ELAPSED/3600)) \
    $(((TOTAL_ELAPSED%3600)/60)) \
    $((TOTAL_ELAPSED%60))
echo "=========================================="

