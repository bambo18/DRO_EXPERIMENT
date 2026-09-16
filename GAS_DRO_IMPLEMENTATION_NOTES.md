# GAS-DRO 5D Baseline 구현 및 디버깅 기록

## 1. 목적

본 구현은 공식 GAS-DRO 코드를 현재 5차원 synthetic regression 실험에 맞게 적용한 baseline이다.

- Joint endogenous vector: `Z = [X1, X2, X3, X4, Y]`
- Predictor: `4 -> 64 -> 64 -> 1` MLP
- GAS-DRO는 **SCM 제약 없이 joint endogenous space에서 직접 adversarial distribution을 생성**한다.
- 본 baseline은 추후 제안 방법(Ours)의 causal/exogenous factorization과 비교하기 위한 기준선이다.

중요 원칙:

> 공식 GAS-DRO의 hyperparameter는 임의로 변경하지 않는다.  
> 변경은 Carbon/time-series 표현을 5D vector regression task로 옮기기 위해 필요한 adapter 수준으로만 제한한다.

---

## 2. 최종 유지할 핵심 파일

권장 구조:

```text
DRO_EXPERIMENT/
├─ configs/
│  └─ gas_dro_full.py
│
├─ data/
│  ├─ synthetic_scm.py
│  └─ datasets.py
│
├─ methods/
│  └─ gas_dro/
│     ├─ __init__.py
│     ├─ gas_dro.py
│     └─ vector_diffusion.py
│
├─ models/
│  └─ mlp.py
│
├─ experiments/
│  ├─ preflight_official_gas_dro.py
│  └─ run_gas_dro.py
│
├─ docs/
│  └─ GAS_DRO_IMPLEMENTATION_NOTES.md
│
├─ results/
├─ .gitignore
└─ README.md
```

`preflight_official_gas_dro.py`는 삭제하지 않는 것을 권장한다.  
최종 코드 변경 후 official adapter가 여전히 정상인지 확인할 수 있는 integration/regression test 역할을 한다.

---

## 3. 공식 GAS-DRO 설정

### Nominal diffusion

```text
T                  = 500
beta_start         = 0.1
beta_end           = 0.5
batch_size         = 64
learning_rate      = 1e-4
training iteration = 7000
optimizer          = Adam
gradient clipping  = None
```

### GAS-DRO outer / inner optimization

```text
Outer iteration            = 15
Generator inner iteration  = 10
Predictor inner iteration  = 2

Batch size                 = 64
BATCH_REPEAT               = 4

Generator LR               = 1e-5
Predictor LR               = 1e-5

PPO clip                    = 0.4

Initial mu                  = 1.0
ETA                         = 0.1
JSM budget                  = 0.015

ADJUST_TIMESTEPS            = 15

StepLR step_size            = 2
StepLR gamma                = 0.05

P_S0                        = 0
Generator gradient clipping = None
```

이 값들은 최종 baseline에서 성능이 좋지 않더라도 임의로 수정하지 않는다.

---

## 4. Task-specific adaptation

공식 구현에서 변경한 것은 task 표현과 모델 interface다.

### 공식 GAS-DRO

```text
Carbon time-series
-> 28x28 형태 diffusion representation
-> DeepLSTM predictor
```

### 현재 실험

```text
[X1, X2, X3, X4, Y]
-> 5D vector diffusion
-> common RegressionMLP
```

유지하는 GAS-DRO 핵심 구조:

```text
nominal diffusion theta_0
        ↓
fixed nominal trajectory
        ↓
a0
        ↓
adversarial theta
        ↓
r_theta
        ↓
PPO objective + mu * JSM
        ↓
adversarial S_theta generation
        ↓
predictor update
```

SCM은 GAS-DRO generator에 사용하지 않는다.

---

## 5. 구현 중 발견한 문제와 해결

### 문제 1. `ModuleNotFoundError: No module named 'methods'`

하위 폴더의 파일을 직접 실행하면 프로젝트 루트가 import 경로로 잡히지 않았다.

```powershell
python .\experiments\xxx.py
```

대신 프로젝트 루트에서 module 방식으로 실행한다.

```powershell
python -m experiments.preflight_official_gas_dro
python -m configs.gas_dro_full
```

---

### 문제 2. 공식 diffusion schedule에서 `alpha_bar = 0`

공식 설정:

```text
T = 500
beta = 0.1 -> 0.5
```

에서 `alpha_bar_t = product(alpha_1 ... alpha_t)`를 float32로 계산하면 뒤쪽 timestep에서 underflow가 발생했다.

실제 확인:

```text
alpha_bar min : 0.0
zero count    : 138
```

이 현상 자체는 공식 schedule을 바꿔야 한다는 뜻이 아니다.

---

### 문제 3. generic DDPM reverse 구현에서 Inf 발생

초기 vector diffusion 구현은 일반적인 DDPM 방식으로 `1 / sqrt(alpha_bar_t)`를 사용해 `x_0`를 복원했다.

공식 schedule에서는 일부 `alpha_bar_t = 0`이므로:

```text
1 / 0 -> Inf
```

가 발생했다.

#### 해결

공식 GAS-DRO/DDPM reverse mean 형태로 변경했다.

```text
mu_theta(x_t, t)
=
sqrt(1 / alpha_t)
*
[
  x_t
  -
  beta_t / sqrt(1 - alpha_bar_t)
  * epsilon_theta(x_t,t)
]
```

이 방식에서는 `alpha_bar_t = 0`이어도 `sqrt(1 - alpha_bar_t) = 1`이므로 reverse mean을 finite하게 계산할 수 있다.

공식 `T`, `beta` 값은 변경하지 않았다.

---

### 문제 4. 학습하지 않은 diffusion의 500-step sampling overflow

공식 reverse 식으로 변경한 뒤에도 랜덤 초기화된 denoiser로 500 reverse step을 수행했을 때 값이 약 `1e38`까지 커진 후 Inf가 발생했다.

이는 공식 schedule의 오류가 아니라, 학습되지 않은 `epsilon_theta`가 500 step 동안 누적되며 발생한 현상이었다.

공식 설정으로 nominal diffusion을 7000 optimizer iterations 학습한 뒤 sampling:

```text
Shape      : torch.Size([32, 5])
All finite : True
Min        : -1.7661
Max        :  2.4373
Mean       :  0.0293
Std        :  0.7186
Runtime    : 84.18 sec
```

따라서 학습된 diffusion에서는 full 500-step sampling이 정상 동작함을 확인했다.

---

### 문제 5. `7000 iteration`과 `7000 epoch` 구분

공식 diffusion training은 `7000 optimizer updates`이지 `7000 full dataset epochs`가 아니다.

그래서 `train_vector_diffusion_steps()`를 추가해 fixed optimizer iteration 방식으로 구현했다.

```text
optimizer = Adam
batch_size = 64
lr = 1e-4
total optimizer iterations = 7000
```

기존 epoch 기반 helper는 과거 smoke/diagnostic 용도로만 남겨둘 수 있다.

---

### 문제 6. `BATCH_REPEAT=4` 의미

초기 구현에서는 `generator_batches_per_epoch = 4`로 사용했지만, 공식 GAS-DRO의 `4`는 고정 inner batch 개수가 아니라 `BATCH_REPEAT`이다.

5D vector task에 대응한 의미:

```text
s0_iterations
=
ceil(N / batch_size)
*
BATCH_REPEAT
```

Preflight:

```text
N = 64
batch_size = 64
BATCH_REPEAT = 4

s0_iterations
=
ceil(64 / 64) * 4
=
4
```

실제 결과:

```text
s0 iterations: 4
[PASS] ceil(64 / 64) * 4 = 4
```

---

### 문제 7. trajectory / a0 / r_theta indexing 검증

`theta = theta_0` 상태에서는 반드시:

```text
a_theta = a0
a_diff = 0
r_theta = exp(0) = 1
```

이 되어야 한다.

Preflight 결과:

```text
ratio mean : 1.0
ratio min  : 1.0
ratio max  : 1.0
max |ratio - 1| : 0.0
```

Generator를 1회 update한 뒤:

```text
ratio mean : 1.0028558
ratio min  : 0.9886762
ratio max  : 1.0224935
```

로 실제 theta 변화가 ratio에 반영되었다.

---

## 6. 최종 Preflight 결과

실행:

```powershell
python -m experiments.preflight_official_gas_dro
```

결과:

```text
Official hyperparameters modified : NO
Initial r_theta ~= 1             : YES
Generator update completed        : YES
S_theta finite                    : YES
Predictor update completed        : YES
Runtime                           : 85.78 sec
```

세부 결과:

```text
Temporary joint shape       : [64, 5]

Nominal diffusion
- T                         : 500
- iterations                : 7000
- final loss                : 0.0029204
- all finite                : True

Reference
- s0 iterations             : 4
- reference_joint           : [256, 5]
- reference_trajectory      : [256, 15, 5]
- a0                        : [256, 15]

Initial ratio
- mean                      : 1.0
- min                       : 1.0
- max                       : 1.0

One generator update
- PPO                       : 0.4671874
- JSM                       : 0.4033902
- objective                 : -0.0637972

Post-update ratio
- mean                      : 1.0028558
- min                       : 0.9886762
- max                       : 1.0224935

Generated S_theta
- shape                     : [64, 5]
- finite                    : True
- min                       : -2.5111
- max                       : 2.3143

Predictor update
- epoch 1 MSE               : 0.709553
- epoch 2 MSE               : 0.709235
```

이 결과는 성능 평가가 아니라 **GAS-DRO 5D adapter의 integration 검증 결과**다.

---

## 7. 과거 diagnostic 실험

다음 실험은 구현 문제를 찾기 위해 사용했다.

- budget sweep
- fixed mu sweep
- no-gradient-clipping / large-mu sweep
- gradient diagnostic
- temporary OOD pilot

이 결과들은 최종 GAS-DRO hyperparameter 선택에 사용하지 않는다.

특히 `mu=40`, `mu=50` 등은 gradient scale을 이해하기 위한 diagnostic일 뿐이며 최종 baseline에는 적용하지 않는다.

최종 baseline:

```text
mu = 1
eta = 0.1
budget = 0.015
```

을 유지한다.

---

## 8. 현재 삭제/정리 권장 파일

legacy diagnostic:

```text
experiments/pilot_budget_sweep.py
experiments/pilot_gradient_diagnostic.py
experiments/pilot_mu_no_clip.py
experiments/pilot_mu_sweep.py
experiments/pilot_ood_gas_dro.py

experiments/smoke_gas_dro.py
experiments/smoke_gas_dro_loop.py
experiments/smoke_pipeline.py
experiments/smoke_official_diffusion.py

configs/gas_dro_smoke.py
```

`smoke_official_diffusion.py`는 현재 `preflight_official_gas_dro.py`가 nominal diffusion + 전체 GAS-DRO 연결을 모두 검증하므로 중복이다.

유지 권장:

```text
configs/gas_dro_full.py

methods/gas_dro/__init__.py
methods/gas_dro/gas_dro.py
methods/gas_dro/vector_diffusion.py

models/mlp.py

experiments/preflight_official_gas_dro.py
experiments/run_gas_dro.py

data/synthetic_scm.py
data/datasets.py
```

`methods/gas_dro/config.py`가 비어 있고 어느 곳에서도 import하지 않는다면 삭제 가능하다.

`experiments/run_gas_dro.py`는 최종 팀 코드 연결 시 사용할 예정이므로, 내용이 과거 smoke용이면 나중에 최종 runner로 덮어쓴다.

---

## 9. `.gitignore`

```gitignore
# Official GAS-DRO reference
gas_dro_official/
gas_dro_official.zip

# Generated datasets
data/generated/

# Model checkpoints
checkpoints/
*.pt
*.pth
*.ckpt

# Raw experimental outputs
results/raw/

# Python cache
__pycache__/
*.pyc

# VSCode / IDE
.vscode/
```

---

## 10. 다음 단계

GAS-DRO 내부 구현 자체는 현재 상태에서 더 수정하지 않는다.

팀으로부터 다음을 받아 연결한다.

1. 최종 nominal synthetic data
2. 공통 predictor protocol
3. Validation OOD environments
4. Held-out Mild / Moderate / Strong OOD evaluation
5. 공통 5 seeds

최종 비교에서 GAS-DRO는 SCM 정보를 adversarial generation에 사용하지 않는다.

---

## 11. 현재 상태 요약

```text
Official GAS-DRO reference 확인
        ✅

공식 hyperparameter 고정
        ✅

5D vector diffusion adapter
        ✅

T=500 / beta=0.1→0.5 수치 문제 해결
        ✅

7000-step nominal diffusion
        ✅

500-step reverse sampling
        ✅

fixed trajectory
        ✅

a0
        ✅

r_theta
        ✅

PPO
        ✅

JSM
        ✅

generator update
        ✅

S_theta generation
        ✅

predictor update
        ✅

official integration preflight
        ✅

--------------------------------

남은 작업

팀 최종 데이터 연결
        ↓

공통 predictor 초기 학습 연결
        ↓

validation OOD
        ↓

official full GAS-DRO
        ↓

5-seed held-out OOD evaluation
```
