"""
Official GAS-DRO configuration adapted to our 5D regression task.

IMPORTANT
---------
Values under "Official GAS-DRO hyperparameters" come directly from
the official GAS-DRO repository and must not be arbitrarily changed
for the final baseline.

Task-specific changes are explicitly separated below.

Official repository:
https://github.com/CIGLAB-Houston/GAS-DRO
"""

import sys
from pathlib import Path


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from methods.gas_dro.gas_dro import GasDROConfig


# ============================================================
# 1. Task-specific adapter settings
#
# These are NOT GAS-DRO hyperparameters.
#
# Original GAS-DRO:
#     Carbon time-series -> 28 x 28 representation
#
# Our task:
#     Z = [X1, X2, X3, X4, Y]
# ============================================================

DATA_DIM = 5

PREDICTOR_INPUT_DIM = 4
PREDICTOR_OUTPUT_DIM = 1
PREDICTOR_HIDDEN_DIM = 64


# ============================================================
# 2. Official diffusion configuration
#
# From official GAS-DRO main.py:
#
# T          = 500
# BETA_1     = 0.1
# BETA_T     = 0.5
# BATCH_SIZE = 64
# LR         = 1e-4
# ITERATION  = 7000
# ============================================================

DIFFUSION_CONFIG = {

    "timesteps": 500,

    "beta_start": 1e-1,

    "beta_end": 5e-1,

    "batch_size": 64,

    "lr": 1e-4,

    # IMPORTANT:
    # Official implementation uses optimizer iterations,
    # not "epochs over the whole dataset".
    "iterations": 7000,

    "selected_iteration": 7000,
}


# ============================================================
# 3. Official GAS-DRO hyperparameters
#
# main.py:
#
# PUBLIC
# -------
# BATCH_SIZE       = 64
# DISCOUNT_FACTOR  = 0.05
# STEP_SIZE        = 2
# PPO_CLIP         = 0.4
#
# DRO
# ---
# ITERATION        = 15
# ETA              = 0.1
# MU               = 1
# BUDGET           = 0.015
# LR               = 1e-5
# ADJUST_TIMESTEPS = 15
# P_S0             = 0
#
# DIFFUSION TRAIN
# ----------------
# BATCH_REPEAT     = 4
# ITERATION        = 10
#
# ML TRAIN
# --------
# LR               = 1e-5
# ITERATION        = 2
# ============================================================

GAS_DRO_CONFIG = GasDROConfig(
    outer_epochs=15,
    generator_inner_epochs=10,
    predictor_inner_epochs=2,

    batch_size=64,
    batch_repeat=4,

    generator_lr=1e-5,
    predictor_lr=1e-5,

    ppo_clip=0.4,

    mu=1.0,
    eta=0.1,
    budget=0.015,

    adjust_timesteps=15,

    step_size=2,
    discount_factor=0.05,

    p_s0=0.0,

    grad_clip=None,

    verbose=True,
)


# ============================================================
# 4. Official GAS-DRO constants kept separately
#
# Keeping these explicit prevents accidental reinterpretation.
# ============================================================

OFFICIAL_CONSTANTS = {

    "BATCH_REPEAT": 4,

    "P_S0": 0,

    "ML_SAVE_EVERY": 2,

    "DFU_SAVE_EVERY": 2,
}


# ============================================================
# 5. Print config
# ============================================================

if __name__ == "__main__":

    print("==========================================")
    print("OFFICIAL GAS-DRO FULL CONFIG")
    print("==========================================")

    print("\n[Task adapter]")
    print(f"Joint dimension     : {DATA_DIM}")
    print(
        f"Predictor            : "
        f"{PREDICTOR_INPUT_DIM}"
        f" -> {PREDICTOR_HIDDEN_DIM}"
        f" -> {PREDICTOR_HIDDEN_DIM}"
        f" -> {PREDICTOR_OUTPUT_DIM}"
    )

    print("\n[Official diffusion]")
    for key, value in DIFFUSION_CONFIG.items():
        print(
            f"{key:20s}: {value}"
        )

    print("\n[Official GAS-DRO]")
    print(GAS_DRO_CONFIG)

    print("\n[Other official constants]")
    for key, value in OFFICIAL_CONSTANTS.items():
        print(
            f"{key:20s}: {value}"
        )

    print("==========================================")