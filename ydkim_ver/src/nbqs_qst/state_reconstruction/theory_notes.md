# State reconstruction — 이론적 기반과 구현 설계

> **AI 사용 고지 및 검증 상태**
>
> 이 문서는 2026-08-17에 Codex의 자료 조사와 초안 작성을 바탕으로
> 생성되었다. 아래 수식, 문헌 해석, 구현 권고는 **독립 검증 전이며
> verified로 표시해서는 안 된다**. 구현 전 원문 논문과 수식을 사람이 다시
> 확인하고, 구현 후에는 독립 테스트로 검증해야 한다.

이 문서는 [`challenge_goal_document.md`](../../../docs/challenge_goal_document.md)의
state reconstruction milestone을 현재의
[`MeasurementDataset`](../measurement_generation/pauli_measurement.py)과 연결한다.
목표는 다음 두 필수 기준선을 정확히 정의하는 것이다.

1. Pauli 선형 역산(linear inversion)
2. 유한-shot count의 multinomial likelihood를 사용하는 물리적 MLE

추가 비교 대상으로 projected least squares(PLS), projected-gradient MLE,
diluted `RρR`를 다룬다. 이 문서의 범위는 이론과 구현 설계이며, 아직 구현
완료 또는 성능 검증을 뜻하지 않는다.

---

## 1. 프로젝트 요구사항으로부터 고정되는 문제

`n` qubit에 대해 Hilbert 공간 차원을 `d = 2^n`, 전체 측정 setting 수를
`S = 3^n`이라 하자. 각 setting은

$$
b=(b_0,\ldots,b_{n-1})\in\{X,Y,Z\}^n
$$

이고, outcome은 bit string

$$
x=(x_0,\ldots,x_{n-1})\in\{0,1\}^n
$$

이다. 프로젝트 규약상 qubit 0은 최상위 비트(MSB)이며 문자열과 setting은
lexicographic 순서이다. outcome bit `0`은 해당 Pauli의 고윳값 `+1`, bit
`1`은 `-1`을 뜻한다.

한 qubit의 Pauli 고유 projector는

$$
\Pi_{a,x}=\frac{I+(-1)^x\sigma_a}{2},\qquad a\in\{X,Y,Z\},
$$

이고 전체 setting의 outcome effect는

$$
E_{b,x}=\bigotimes_{q=0}^{n-1}\Pi_{b_q,x_q}.
$$

Born 확률과 count model은

$$
p_{b,x}(\rho)=\operatorname{Tr}(E_{b,x}\rho),
$$

$$
(c_{b,0},\ldots,c_{b,d-1})
\sim\operatorname{Multinomial}(N;p_{b,0},\ldots,p_{b,d-1})
$$

이다. 현재 measurement 모듈의 `counts[i]`가 바로 `c_{b,x}`이고, 모든
setting에 같은 `N = shots_per_setting`을 사용한다. **reconstruction에서
가우시안 오차를 새로 더하거나 multinomial count를 가우시안 데이터로
바꾸지 않는다.**

모든 `XYZ` tensor-product setting을 측정하므로 측정은 informationally
complete하다. 따라서 무한 shot에서는 임의의 `d × d` density matrix를
식별할 수 있다. 다만 finite shot에서는 frequency가 정확히 하나의 물리적
density matrix와 일치하지 않는 것이 일반적이다.

### density matrix가 만족해야 할 조건

$$
\rho=\rho^\dagger,\qquad \rho\succeq0,\qquad
\operatorname{Tr}(\rho)=1.
$$

Hermiticity와 trace one만 만족하는 선형 추정치는 음의 고윳값을 가질 수
있다. Hradil은 inversion이 positivity를 보장하지 않는 문제를 MLE 도입의
핵심 동기로 제시했다
([Hradil, 1997](https://doi.org/10.1103/PhysRevA.55.R1561)).

---

## 2. Pauli operator basis와 선형 역산

`n`-qubit Pauli string 집합

$$
\mathcal P_n=\{I,X,Y,Z\}^{\otimes n}
$$

은 Hilbert–Schmidt 내적에서 직교한다.

$$
\operatorname{Tr}(PQ)=d\,\delta_{P,Q}.
$$

따라서 임의의 density matrix는

$$
\rho=\frac1d\sum_{P\in\mathcal P_n}r_P P,
\qquad r_P=\operatorname{Tr}(P\rho)
$$

로 유일하게 전개된다. 이 프로젝트에서는 일반적인 measurement matrix
`A`와 pseudoinverse를 실제로 만들 필요 없이, Pauli basis의 직교성을
이용해 닫힌 형태의 역산을 사용할 수 있다.

### 2.1 count에서 Pauli expectation 추정

Pauli string `P`의 support를

$$
\operatorname{supp}(P)=\{q:P_q\ne I\}
$$

라 하자. `P`와 호환되는 setting은 support 위에서 `b_q=P_q`인 setting이다.
한 outcome이 주는 `P`의 고윳값은

$$
\lambda_P(x)=\prod_{q\in\operatorname{supp}(P)}(-1)^{x_q}.
$$

`P`에 identity가 `k`개 있으면 호환 setting은 `3^k`개이다. 현재 measurement
모듈과 동일하게 모든 호환 setting을 pooling하면

$$
\widehat r_P=
\frac{1}{N\,3^k}
\sum_{b\sim P}\sum_x c_{b,x}\lambda_P(x),
\qquad P\ne I^{\otimes n},
$$

$$
\widehat r_{I^{\otimes n}}=1.
$$

이는 unbiased estimator이고, 서로 독립인 호환 setting의 shot을 모두
사용한다. 이에 따른 선형 추정치는

$$
\boxed{
\widehat\rho_{\mathrm{LI}}
=\frac1d\sum_{P\in\mathcal P_n}\widehat r_P P
}
$$

이다. `expectations_from_dataset(dataset)`의 결과를 그대로 소비할 수 있다.
James 등은 qubit tomography에서 선형 reconstruction과 positivity를 보장하는
likelihood reconstruction을 함께 상세히 전개했다
([James et al., 2001](https://doi.org/10.1103/PhysRevA.64.052312)).

### 2.2 장점과 한계

- 장점: 닫힌 형태이고 빠르며, expectation과 density-matrix element에 대해
  unbiased 기준선을 제공한다.
- 항상 만족: trace one, 이론상 Hermiticity. 구현에서는 round-off 제거를
  위해 마지막에 `(ρ + ρ†)/2`를 적용할 수 있다.
- 보장하지 않음: positive semidefinite. 음의 고윳값은 finite-shot 데이터의
  정상적인 결과이지, 임의로 숨겨야 할 오류가 아니다.
- `ρ_LI`가 비물리적이면 purity가 1보다 커지거나 fidelity 계산이 불안정할
  수 있다. 따라서 LI 평가에서는 최소 고윳값과 negativity도 함께 기록한다.
- 이 프로젝트의 Pauli basis에서는 `A⁺f`를 명시적으로 구성하는 방식보다
  위 식이 더 단순하다. 일반 POVM API를 추가할 때만 design matrix/dual frame
  형태를 별도로 고려한다.

### 2.3 Y 부호 회귀 규칙

elementwise contraction으로 trace를 계산할 때는

$$
\operatorname{Tr}(P\rho)=\sum_{ij}P_{ij}\rho_{ji}
=\sum_{ij}P_{ij}(\rho^T)_{ij}
$$

이므로 `ρ`를 **전치**해야 한다. `Y^T=-Y`이기 때문에 `sum(P * ρ)`는 Y가
홀수 번 포함된 항의 부호를 뒤집는다. reconstruction의 forward model,
gradient, 검증 코드에도 같은 규칙이 적용된다. 가능하면 `trace(E @ ρ)`를
사용하고, elementwise 최적화 시에만 명시적 transpose를 사용한다.

---

## 3. 물리 상태로의 projection과 PLS

Hermitian trace-one 행렬 `H`를 Frobenius norm에서 density-matrix 집합에
투영하면 projected least squares 기준선을 만들 수 있다. 먼저

$$
H=V\operatorname{diag}(\lambda)V^\dagger
$$

로 고유분해하고, 고윳값 벡터를 probability simplex에 투영한다.

$$
\mu_i=\max(\lambda_i-\tau,0),\qquad \sum_i\mu_i=1,
$$

여기서 `τ`는 합이 1이 되게 정한다. 그 다음

$$
\widehat\rho_{\mathrm{PLS}}
=V\operatorname{diag}(\mu)V^\dagger
$$

이다. PLS는 빠른 물리 추정치이며 Pauli measurement에 대한 non-asymptotic
오차 보장이 연구되어 있다
([Guta et al., 2020](https://arxiv.org/abs/1809.11162)).

그러나 이 프로젝트에서 PLS를 MLE라고 부르면 안 된다. Smolin 등의 단일
projection 결과는 **complete orthonormal operator basis와 additive Gaussian
noise**라는 특정 모형에서의 결과다
([Smolin et al., 2012](https://doi.org/10.1103/PhysRevLett.108.070502)).
현재 프로젝트의 데이터는 setting별 multinomial count이므로 PLS는 별도
baseline 또는 MLE 초기값일 뿐이다.

---

## 4. Multinomial maximum-likelihood estimation

모든 setting이 독립적으로 측정되므로 likelihood는

$$
\mathcal L(\rho)
=\prod_b\left[
\frac{N!}{\prod_x c_{b,x}!}
\prod_x p_{b,x}(\rho)^{c_{b,x}}
\right].
$$

`ρ`와 무관한 factorial 항을 버리면 negative log-likelihood(NLL)는

$$
\boxed{
\mathcal C(\rho)
=-\sum_b\sum_{x:c_{b,x}>0}c_{b,x}
\log\operatorname{Tr}(E_{b,x}\rho)
}
$$

이고 MLE는

$$
\widehat\rho_{\mathrm{MLE}}
=\underset{\rho\succeq0,\,\operatorname{Tr}\rho=1}{\arg\min}
\;\mathcal C(\rho)
$$

이다. `C/(SN)`처럼 양의 상수로 objective를 rescale해도 최적점은 같다.
rescale은 서로 다른 qubit 수와 shot 수에서 step size를 비교하기 쉽게 한다.

`-log`는 convex이고 `p_{b,x}(ρ)`는 `ρ`에 대한 선형 함수이므로 `C(ρ)`는
convex하다. density-matrix 집합도 convex하므로 **`ρ`를 직접 최적화하는
문제에는 가짜 local minimum이 없다.** 다만 factor parameterization을
도입하면 parameter 공간의 문제는 non-convex가 된다.

Hradil의 MLE는 positivity 위반을 피하기 위한 표준 출발점이며
([Hradil, 1997](https://doi.org/10.1103/PhysRevA.55.R1561)), 후속 연구는
안정적 likelihood 증가를 위한 diluted iteration을 제안했다
([Řeháček et al., 2007](https://doi.org/10.1103/PhysRevA.75.042108)).

### 4.1 zero count와 zero probability

- `c=0`인 항은 수학적으로 `0 log p = 0`이므로 objective 합에서 제외한다.
- `c>0`인데 `p=0`이면 NLL은 `+∞`이다. 이를 `log(0)` 직전에만 임의의 큰
  epsilon으로 바꾸면 원래 likelihood가 달라진다.
- 안전한 기본 초기값은 `ρ₀=I/d`이다. 모든 rank-one Pauli outcome에 대해
  `p=1/d>0`이다.
- line search 중 positive-count outcome의 확률이 허용 tolerance 이하가
  되는 candidate는 reject하는 편이 의미가 명확하다. 부득이하게 probability
  floor를 쓰면 그 값과 objective 변경을 API 및 결과에 기록한다.
- round-off로 생긴 작은 허수부는 `real(...)`로 제거할 수 있지만, 큰 허수부나
  음의 확률은 forward-model 오류로 처리해야 한다.

### 4.2 NLL의 density-matrix gradient

Hermitian matrix 공간에서 NLL gradient는

$$
\nabla_\rho\mathcal C(\rho)
=-\sum_{b,x:c_{b,x}>0}
\frac{c_{b,x}}{p_{b,x}(\rho)}E_{b,x}.
$$

objective를 `SN`으로 나누면 gradient도 같은 값으로 나눈다. 각 `E`는
Hermitian이므로 gradient도 Hermitian이다. 이 식을 직접 구현하면 backend별
autograd나 SciPy optimizer가 필요 없다.

---

## 5. MLE를 구현하는 세 가지 방법

### 5.1 Factor/Cholesky parameterization

challenge 문서가 제시한 형태는

$$
\rho(T)=\frac{T^\dagger T}{\operatorname{Tr}(T^\dagger T)}.
$$

이는 모든 iterate에서 PSD와 trace one을 보장한다. `T`를 unrestricted
complex `d × d` factor로 두면 singular `T`를 통해 rank-deficient state도
표현할 수 있다. 반면 positive diagonal을 강제한 strict Cholesky factor만
사용하면 rank-deficient boundary를 정확히 표현하기 어렵다.

장점은 eigenvalue projection이 필요 없고 challenge 설명과 직접 일치한다는
점이다. 단점은 다음과 같다.

- `T`와 `UT`(`U` unitary)가 같은 `ρ`를 나타내는 gauge redundancy가 있다.
- `ρ`에 대해서는 convex인 문제가 `T`에서는 non-convex가 된다.
- backend-neutral 구현에서는 특정 backend의 autograd에 의존할 수 없으므로
  analytic gradient 또는 순수 matrix iteration이 필요하다.
- `T†T`에서 dagger는 `matrix_transpose(T.conj())`이다. transpose만 사용하면
  complex state가 잘못된다.

이 경로를 구현할 경우 arbitrary optimizer를 black box로 호출하기보다,
objective 감소를 확인하는 backtracking과 명시적 convergence diagnostics를
포함해야 한다.

### 5.2 Density matrix에 대한 projected-gradient MLE

직접 제약 최적화를 하면 한 step은

$$
\rho_{k+1}=\mathcal P_{\mathcal D}
\left[\rho_k-\eta_k\nabla\mathcal C(\rho_k)\right]
$$

이다. `P_D`는 3절의 eigenvalue-simplex projection이다. backtracking으로
NLL이 실제 감소하는 step만 받아들인다. projected-gradient 방식은 density
matrix의 convex geometry를 그대로 유지하며 여러 QST 연구에서 빠른 MLE
방법으로 비교되었다
([Shang et al., 2017](https://arxiv.org/abs/1609.07881),
[Bolduc et al., 2017](https://doi.org/10.1038/s41534-017-0043-1)).

하드웨어 독립 구현에 필요한 핵심 연산은 `eigh`, matrix multiply, `sum`,
`log`, `where`, `sort` 정도이며 NumPy/CuPy/JAX/PyTorch가 모두 제공한다.
단, eigenvalue projection은 iteration마다 `O(d^3)` 비용이 든다.

### 5.3 Diluted `RρR`

빈도를 전체 shot 수로 정규화해 `f_{b,x}=c_{b,x}/(SN)`이라 하고

$$
R(\rho)=\sum_{b,x:c_{b,x}>0}
\frac{f_{b,x}}{p_{b,x}(\rho)}E_{b,x}
$$

를 정의할 수 있다. `RρR` 계열은 positive `ρ`에서 시작해 양쪽에 positive
operator를 곱하고 trace를 다시 정규화하므로 physicality를 유지한다.
원래의 full step은 항상 likelihood를 증가시키지 않을 수 있어, diluted
step과 adaptive line search가 중요하다. Řeháček 등의 diluted algorithm은
iteration마다 likelihood 증가와 MLE로의 수렴을 목적으로 설계되었다
([논문](https://doi.org/10.1103/PhysRevA.75.042108)).

이 방식은 manual gradient/eigendecomposition 없이 matrix operation만으로
구현하기 쉬운 장점이 있지만, 문제에 따라 수렴이 느릴 수 있다. 정확한
dilution 식과 line-search 조건은 구현 전에 원 논문의 notation과 이
프로젝트의 `S` normalization을 대조해 다시 유도해야 한다.

### 5.4 이 프로젝트에 대한 1차 권고

| 역할 | 권고 방법 | 이유 |
|---|---|---|
| 필수 baseline | Pauli closed-form LI | 현재 expectation API를 그대로 사용하고 해석이 명확함 |
| 선택적 physical baseline | PLS | 빠른 물리 추정치와 MLE 초기값; 단, MLE로 명명 금지 |
| 필수 MLE의 우선 구현 | projected gradient + backtracking | 정확한 multinomial NLL, convex `ρ` 공간, backend-neutral analytic gradient |
| challenge 문구와 직접 비교 | unrestricted factor `T†T/Tr` MLE | Cholesky식 parameterization의 장단점 및 성능 비교 가능 |
| 대안/확장 | diluted `RρR` | physical iterate와 backend-neutral matrix 연산 |

구현 범위를 작게 유지해야 한다면 **LI + projected-gradient multinomial
MLE**가 가장 명확한 조합이다. factorized MLE는 동일 count와 초기 상태를
사용하는 비교 엔진으로 추가하는 것이 과학적으로 더 유익하다.

---

## 6. 메모리와 계산량: `n=20` 목표의 현실성

full density matrix 자체가 `d²=4^n`개의 complex number를 가진다.
`complex128`에서는 대략

$$
16\cdot4^n\ \text{bytes}
$$

가 필요하다. `n=20`이면 density matrix 하나만 약 16 TiB이다. 따라서
challenge의 `n=20` CPU/GPU MLE benchmark는 **dense full-state MLE로는
실현 불가능**하며, low-rank factor, tensor-network 구조, matrix-free Pauli
연산, 또는 관측량만 추정하는 classical shadows 같은 문제 변경이 필요하다.

또한 모든 effect를 materialize하면 effect 수가 `S d = 6^n`이고 각 effect가
`d²`이므로 메모리가 `24^n`에 비례한다. 구현은 최소한 다음을 지켜야 한다.

- 모든 `E_{b,x}`를 한꺼번에 저장하지 않는다.
- setting 단위로 basis rotation과 probability/gradient contribution을
  계산해 누적한다.
- 가능한 경우 tensor-product 구조로 probability와 gradient를 계산한다.
- dense exhaustive QST benchmark와 structured/approximate method benchmark를
  같은 scaling 주장으로 섞지 않는다.

전체 `3^n` setting을 `N` shot씩 쓰는 현재 protocol의 총 copy 수는
`N_total = N 3^n`이다. plot의 x축이 `shots per setting`인지 `total shots`인지
반드시 표시해야 한다.

---

## 7. 통계적 해석과 평가 지표

### 7.1 LI와 MLE의 bias

LI는 operator coefficient에 대해 unbiased이지만 비물리 상태를 낼 수 있다.
positivity-constrained MLE/least squares는 finite sample에서 일반적으로
biased다. 특히 state-space boundary 근처의 pure/low-rank state에서 이
효과가 중요하다. 관련 연구는 물리 제약을 둔 추정이 fidelity를 체계적으로
낮추거나 entanglement를 높게 추정할 수 있음을 보고했다
([Schwemmer et al., 2015](https://doi.org/10.1103/PhysRevLett.114.080403),
[Silva et al., 2017](https://doi.org/10.1103/PhysRevA.95.022107)).

따라서 “physical estimate이므로 더 정확하다” 또는 “MLE이므로 unbiased다”라고
주장하면 안 된다. state family와 purity, shot 수별 반복 Monte Carlo 실험으로
bias와 variance를 함께 보고해야 한다.

### 7.2 프로젝트 평가량

challenge 문서의 정의를 그대로 사용한다.

$$
\operatorname{purity}(\rho)=\operatorname{Tr}(\rho^2),
$$

$$
F(\rho,\sigma)=
\left(\operatorname{Tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\right)^2,
$$

pure target에 대해서는

$$
F(|\psi\rangle\langle\psi|,\widehat\rho)
=\langle\psi|\widehat\rho|\psi\rangle.
$$

두 pure state vector를 비교할 때만

$$
|\langle\psi|\widehat\psi\rangle|^2
$$

을 overlap으로 쓴다. 추가로 다음 diagnostics가 유용하다.

- trace error `|Tr(ρ)-1|`
- Hermiticity error `||ρ-ρ†||_F`
- minimum eigenvalue
- trace distance `0.5 ||ρ_true-ρ_est||_1`
- held-out 또는 training NLL
- iteration 수, wall time, peak memory, convergence flag

purity는 state 특성이지 단독 reconstruction error가 아니다. true purity와
estimate purity의 차이나 bias로 보고해야 한다.

### 7.3 confidence interval 주의

positivity 때문에 MLE가 state-space boundary에 쌓일 수 있어 저 rank에서
표준 Gaussian/Wilks 근사가 깨질 수 있다
([Scholten & Blume-Kohout, 2018](https://arxiv.org/abs/1609.04385)).
초기 구현에서는 asymptotic error bar를 자동으로 신뢰하기보다, stdlib
`random.Random`으로 measurement dataset 전체를 반복 생성하는 parametric
Monte Carlo를 우선 고려한다. 이 역시 estimator bias를 제거하지 않으므로
반복 수와 절차를 명시해야 한다.

---

## 8. Hardware-agnostic 구현 원칙

reconstruction 코드도 저장소의 최상위 규칙을 그대로 따라야 한다.

1. 입력 backend array에서 `xp = array_namespace(...)`를 얻고 core logic에서
   NumPy를 직접 import하지 않는다.
2. `complex128`을 유지한다. JAX 테스트 전에
   `jax.config.update("jax_enable_x64", True)`를 호출한다.
3. JAX 배열은 immutable이므로 eigenvalue 수정, parameter update, history
   저장에 in-place assignment를 쓰지 않는다.
4. backend optimizer/autograd에 핵심 알고리즘을 묶지 않는다. analytic
   gradient와 공통 matrix operation을 우선한다.
5. counts가 있는 backend/device에 상수, Pauli matrix, basis rotation을 만든다.
6. Python scalar로 매 iteration 값을 꺼내는 것은 GPU sync를 일으킨다.
   line search와 diagnostics에 필요한 sync 횟수를 측정하고 문서화한다.
7. reconstruction 자체는 난수를 필요로 하지 않아야 한다. stochastic
   initialization을 추가한다면 난수는 오직 stdlib `random.Random`을 쓴다.

`MeasurementDataset`은 backend-native count tuple을 이미 보존한다. 첫 count
array로 namespace/device를 감지하고, setting/count ordering이 일치하는지,
각 count가 nonnegative integer인지, setting별 합이 `shots_per_setting`인지
입력 경계에서 검증하는 것이 적절하다.

---

## 9. 권장 API와 모듈 경계

아직 구현 전 설계안이다.

```text
state_reconstruction/
    linear_inversion.py       # dataset/expectations -> rho_LI
    physical_projection.py    # Hermitian + simplex projection, PLS
    maximum_likelihood.py     # multinomial NLL, gradient, PGD MLE
    reconstruction_test/
```

공통 결과 객체에는 최소한 다음을 포함하는 것이 좋다.

```text
rho
method
converged
iterations
objective
objective_history (선택)
diagnostics: trace_error, hermiticity_error, min_eigenvalue
```

projector/effect 생성은 measurement와 reconstruction에서 중복 구현하면 Y
부호, bit ordering, basis convention이 어긋날 위험이 있다. 공개 shared helper로
승격하거나, 두 모듈이 동일한 작은 convention 모듈을 사용하도록 정리하는
편이 안전하다. 단, 현재 measurement의 private helper를 reconstruction에서
직접 import하는 것은 피한다.

---

## 10. 구현 전 테스트 계획

### 10.1 선형 역산

- ideal expectation으로 `|0⟩`, `|+⟩`, `|+y⟩`, maximally mixed state를 정확히
  복원한다.
- `|+y⟩`에서 Y expectation과 imaginary off-diagonal 부호를 확인한다.
- 모든 결과가 trace one, Hermitian인지 확인한다.
- finite-shot LI에서 음의 고윳값이 나오는 deterministic fixture를 보존해
  “자동 positivity clipping” 회귀를 막는다.
- 직접 Pauli expansion과 일반 design-matrix pseudoinverse를 작은 `n`에서
  독립적으로 비교한다.

### 10.2 PLS

- 출력 고윳값이 nonnegative이고 합이 1인지 확인한다.
- 이미 물리적인 입력은 tolerance 안에서 바뀌지 않는다.
- 알려진 작은 eigenvalue vector에 대해 simplex projection을 손계산 값과
  비교한다.

### 10.3 MLE

- 출력이 Hermitian, PSD, trace one이다.
- accepted iteration마다 NLL이 증가하지 않는다.
- 최종 NLL이 `I/d` 초기값보다 작거나 같다.
- analytic gradient를 작은 1-qubit fixture의 central finite difference와
  독립 비교한다.
- noiseless/대규모-shot limit에서 true state로 수렴한다.
- zero count는 안전하고, positive count/zero probability candidate는
  명시적으로 reject된다.
- pure, mixed, maximally mixed target을 모두 포함한다.
- 동일 count dataset은 NumPy/CuPy/JAX/PyTorch에서 tolerance 안의 같은
  density matrix와 objective를 낸다.
- shots 증가에 따른 error 감소를 여러 seed의 평균과 uncertainty로 확인한다.

### 10.4 통합 회귀

- measurement의 `settings[i]`, `counts[i]`, MSB convention을 reconstruction이
  정확히 해석한다.
- basis projector로 다시 계산한 확률이 measurement 모듈의 Born 확률과
  일치한다.
- 모든 공개 모듈에는 같은 이름의 이론 companion `.md`를 둔다.
- AI 생성 코드, 수식, 문서의 범위를 고지하고 독립 검증 완료 전 verified
  표기를 하지 않는다.

---

## 11. 조사 문헌과 이 프로젝트에서의 의미

아래는 2026-08-17에 확인한 핵심 자료이다. DOI/arXiv 링크는 원 논문 또는
저자 공개본으로 연결한다.

1. D. F. V. James, P. G. Kwiat, W. J. Munro, A. G. White,
   “On the Measurement of Qubits,” *Physical Review A* 64, 052312 (2001).
   [DOI](https://doi.org/10.1103/PhysRevA.64.052312) — qubit 선형 역산,
   likelihood reconstruction, 실험적 error analysis의 고전적 참고문헌.
2. Z. Hradil, “Quantum-state estimation,” *Physical Review A* 55,
   R1561 (1997). [DOI](https://doi.org/10.1103/PhysRevA.55.R1561) — inversion의
   positivity 문제와 MLE state estimation의 출발점.
3. J. Řeháček, Z. Hradil, E. Knill, A. I. Lvovsky, “Diluted
   maximum-likelihood algorithm for quantum tomography,” *Physical Review A*
   75, 042108 (2007).
   [DOI](https://doi.org/10.1103/PhysRevA.75.042108) — monotonic likelihood
   증가를 위한 diluted iteration과 adaptive procedure.
4. J. Shang, Z. Zhang, H. K. Ng, “Superfast maximum-likelihood
   reconstruction for quantum tomography,” *Physical Review A* 95, 062336
   (2017). [arXiv](https://arxiv.org/abs/1609.07881) — accelerated
   projected-gradient MLE와 대규모 비교.
5. E. Bolduc, G. C. Knee, E. Gauger, J. Leach, “Projected gradient descent
   algorithms for quantum state tomography,” *npj Quantum Information* 3,
   44 (2017). [Open access](https://doi.org/10.1038/s41534-017-0043-1) —
   density-matrix projection을 포함한 PGD 변형과 알고리즘 비교.
6. M. Guta, J. Kahn, R. Kueng, J. A. Tropp, “Fast state tomography with
   optimal error bounds,” *Journal of Physics A* 53, 204001 (2020).
   [arXiv](https://arxiv.org/abs/1809.11162) — PLS와 Pauli measurement에 대한
   non-asymptotic trace-distance guarantee.
7. J. A. Smolin, J. M. Gambetta, G. Smith, “Efficient Method for Computing
   the Maximum-Likelihood Quantum State from Measurements with Additive
   Gaussian Noise,” *Physical Review Letters* 108, 070502 (2012).
   [DOI](https://doi.org/10.1103/PhysRevLett.108.070502) — eigenvalue simplex
   projection의 유용한 출처이지만, 이 프로젝트의 multinomial noise와는
   likelihood 가정이 다름.
8. C. Schwemmer et al., “Systematic Errors in Current Quantum State
   Tomography Tools,” *Physical Review Letters* 114, 080403 (2015).
   [DOI](https://doi.org/10.1103/PhysRevLett.114.080403) — positivity-constrained
   reconstruction의 fidelity/entanglement bias 경고.
9. G. B. Silva, S. Glancy, H. M. Vasconcelos, “Investigating bias in
   maximum-likelihood quantum-state tomography,” *Physical Review A* 95,
   022107 (2017). [DOI](https://doi.org/10.1103/PhysRevA.95.022107) — finite
   sample MLE bias와 purity/boundary 의존성.
10. T. L. Scholten, R. Blume-Kohout, “Behavior of the maximum likelihood in
    quantum state tomography,” *New Journal of Physics* 20, 023050 (2018).
    [arXiv](https://arxiv.org/abs/1609.04385) — positivity boundary에서 표준
    Wilks 이론이 깨지는 이유.
11. R. Blume-Kohout, “Hedged Maximum Likelihood Quantum State Estimation,”
    *Physical Review Letters* 105, 200504 (2010).
    [DOI](https://doi.org/10.1103/PhysRevLett.105.200504) — zero eigenvalue와
    overconfident prediction을 완화하는 hedged likelihood; 필수 범위 이후의
    확장 후보.
12. Y. S. Teo, *Introduction to Quantum-State Estimation*, World Scientific
    (2015). [Publisher record](https://books.google.com/books?id=yOSiCgAAQBAJ)
    — informational completeness, likelihood/entropy estimation, 실제 QST
    절차를 포괄하는 보조 교재.

---

## 12. 구현 시작 전 결정 체크리스트

- [ ] 필수 MLE를 projected-gradient로 할지, factor `T` 방식도 동시에 구현할지
  결정한다.
- [ ] NLL normalization을 raw counts 또는 `counts/(SN)` 중 하나로 고정하고
  API에 기록한다.
- [ ] probability tolerance와 line-search acceptance rule을 수식으로 고정한다.
- [ ] effect/basis convention을 measurement와 공유하는 모듈 경계를 결정한다.
- [ ] dense QST의 최대 `n`과 구조화된 확장 benchmark를 분리한다.
- [ ] 독립 reviewer가 이 문서의 수식과 인용을 원문에 대조한다.
- [ ] 검토 후에도 코드 테스트와 수치 검증이 끝나기 전에는 verified 표기를
  하지 않는다.
