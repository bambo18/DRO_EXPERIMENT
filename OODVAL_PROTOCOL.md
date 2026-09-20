# OOD Validation Selection Protocol

This folder contains the final-selection scripts for ERM and GAS-DRO.

## Fixed rules

1. GAS-DRO seed `s` starts from ERM seed `s`.
2. Training gradients use only the nominal training set.
3. OOD validation is used only for hyperparameter/checkpoint/model selection.
4. OOD test is never accepted by the tuning scripts.
5. Predictor normalization uses nominal TRAIN statistics only.
6. Hyperparameter selection score:
   - for each seed: worst OOD-validation environment MSE
   - across seeds: mean of those worst MSEs
   - choose the configuration with the smallest score
7. Seeds: 0,1,2,3,4.
8. Final reporting uses mean ± sample standard deviation.

## ERM

`run_erm_ood_select.py` performs:
- grid training on train.csv only
- epoch/checkpoint selection by worst OOD-validation MSE
- HP selection by mean across seeds of each seed's worst OOD-validation MSE
- copies selected seed checkpoints to `checkpoints/final_oodval/`

Example:

```powershell
python .\experiments\run_erm_ood_select.py `
  --train-csv ".\data\nominal\train.csv" `
  --val-csv ".\data\ood\validation\mild\env01.csv" ".\data\ood\validation\moderate\env01.csv" ".\data\ood\validation\strong\env01.csv" `
  --verbose
```

Pass every OOD-validation environment after `--val-csv`, not only one per strength.

## GAS-DRO

Step 1: train candidate configurations with `run_gas_dro_tunable.py`.

Each candidate must be run for all five seeds. Example official config:

```powershell
python .\experiments\run_gas_dro_tunable.py `
  --seed 0 `
  --train-csv ".\data\nominal\train.csv" `
  --run-tag "official" `
  --budget 0.015 `
  --eta 0.1 `
  --generator-lr 0.00001 `
  --predictor-lr 0.00001 `
  --ppo-clip 0.4
```

Repeat for seeds 0..4 and for every pre-declared GAS-DRO HP candidate.

Step 2: select using OOD validation only:

```powershell
python .\experiments\select_gas_dro_ood.py `
  --val-csv ".\data\ood\validation\mild\env01.csv" ".\data\ood\validation\moderate\env01.csv" ".\data\ood\validation\strong\env01.csv"
```

The selector ignores incomplete configs that do not have exactly the required five seeds.

## Final OOD test

Only after ERM/GAS-DRO selections are locked, evaluate the selected checkpoints with `evaluate_ood.py`.

ERM:

```powershell
python .\experiments\evaluate_ood.py `
  --method erm `
  --split test `
  --data-dir ".\data\ood" `
  --checkpoint-dir ".\checkpoints\final_oodval"
```

GAS-DRO:

```powershell
python .\experiments\evaluate_ood.py `
  --method gas_dro `
  --split test `
  --data-dir ".\data\ood" `
  --checkpoint-dir ".\checkpoints\final_oodval"
```
