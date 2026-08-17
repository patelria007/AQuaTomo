# State Generation: 이론적 기반과 구현 결정 기록

> **AI 사용 고지:** 이 문서는 2026-08-17에 OpenAI Codex의 도움으로
> 조사·작성되었다. 포함된 설명, 수식, 문헌 해석은 독립적인 전문가 검토와
> 수치 검증을 거치기 전까지 **미검증(unverified)** 상태이며, verified로
> 표시해서는 안 된다.

이 문서는 [`challenge_goal_document.md`](../../../docs/challenge_goal_document.md)의
state-generation milestone을 실제 구현으로 옮기기 위한 이론적 기준을
정리한다. 조사 범위는 다음과 같다.

- 각 qubit이 Bloch sphere에 균일하게 분포하는 random product pure state
- 전체 Hilbert space에서 균일한 Haar-random pure state
- rank와 purity가 달라지는 random mixed state
- tomography 비교에 유용한 구조적 기준 상태
- 프로젝트의 Array API, 난수 재현성, qubit-order 규칙에 맞는 구현 방법

---

## 1. 결론: 권장할 상태군

`n` qubit의 Hilbert-space dimension을 `d = 2^n`이라 하자. 첫 구현은 아래
상태군을 서로 다른 생성기로 제공하는 것이 좋다.

| 상태군 | 얽힘 | rank | purity | 주된 비교 목적 |
|---|---:|---:|---:|---|
| Local-Haar product pure | 없음 | 1 | 1 | 비얽힘 baseline |
| Global Haar pure | 일반적으로 큼 | 1 | 1 | 같은 purity에서 entanglement/복잡도 비교 |
| Induced mixed `(d, K)` | 일반적으로 존재 | `min(d,K)` (거의 확실히) | 표본마다 변동 | 자연스러운 random-matrix ensemble, rank 비교 |
| Depolarized Haar, target `gamma` | 일반적으로 존재 | 보통 `d` | 정확히 `gamma` | purity를 통제한 실험 |
| GHZ / W | 구조적 multipartite | 1 | 1 | 알려진 진폭·얽힘 구조의 회귀 및 해석 |

핵심 원칙은 **분포 이름과 파라미터를 결과 metadata에 기록하는 것**이다.
특히 mixed state에는 pure state의 Haar measure와 같은 유일한 "uniform"
measure가 없다. 따라서 단순히 `random_mixed_state`라고 부르면서 분포를
숨기지 말고, 최소한 `ensemble="induced"`, `K`, `seed`를 기록해야 한다
[[Życzkowski & Sommers 2001](https://arxiv.org/abs/quant-ph/0012101)].

---

## 2. 공통 정의와 물리적 조건

### 2.1 Pure state와 density matrix

정규화된 상태 벡터 `|psi>`는

$$
\langle\psi|\psi\rangle = 1
$$

을 만족한다. 이에 대응하는 pure-state density matrix는

$$
\rho = |\psi\rangle\langle\psi|
$$

이다. 모든 유효한 density matrix는 다음 조건을 만족해야 한다.

1. Hermitian: $\rho=\rho^\dagger$
2. Positive semidefinite: $\rho\succeq 0$
3. Unit trace: $\operatorname{Tr}\rho=1$

Purity는

$$
\gamma(\rho)=\operatorname{Tr}(\rho^2)
$$

이며 `d`차원 상태에서는

$$
\frac{1}{d}\leq\gamma\leq 1.
$$

왼쪽 등호는 maximally mixed state `I/d`, 오른쪽 등호는 pure state에서
성립한다. Challenge에서 사용하는 fidelity와 overlap 정의는
[`challenge_goal_document.md`](../../../docs/challenge_goal_document.md#41-state-generation)를
그대로 따른다.

### 2.2 Tensor-product와 bit 순서

프로젝트 규칙상 qubit 0은 최상위 비트(most significant bit)이다. 따라서

$$
|q_0q_1\cdots q_{n-1}\rangle
=|q_0\rangle\otimes|q_1\rangle\otimes\cdots\otimes|q_{n-1}\rangle
$$

순서로 Kronecker product를 수행하고, computational-basis index는

$$
x=\sum_{j=0}^{n-1}q_j2^{n-1-j}
$$

로 해석한다. State generation과 measurement generation이 이 규칙을
공유하지 않으면 GHZ, W 및 product-state의 qubit별 검증이 뒤집힌다.

---

## 3. Random product pure states

### 3.1 정의

Random product state는 각 qubit 상태를 독립적으로 뽑아 tensor product한
상태이다.

$$
|\Psi_{\mathrm{prod}}\rangle
=\bigotimes_{j=0}^{n-1}|\psi_j\rangle.
$$

각 `|psi_j>`는 한 qubit의 Haar measure, 즉 Bloch sphere의 surface-area
measure에 균일해야 한다. 각 qubit은 순수하지만 qubit 사이의 entanglement는
없으며, 전체 상태도 purity 1이다.

한 qubit의 각도 표현은

$$
|\psi_j\rangle
=\cos\frac{\theta_j}{2}|0\rangle
+e^{i\phi_j}\sin\frac{\theta_j}{2}|1\rangle.
$$

균일한 Bloch sphere를 위해서는

$$
\phi_j\sim U[0,2\pi),\qquad
z_j=\cos\theta_j\sim U[-1,1]
$$

이어야 한다. **`theta ~ U[0, pi]`로 직접 뽑으면 극점에 과도한 질량을 주므로
균일하지 않다.**

### 3.2 권장 생성법

각 qubit마다 독립인 complex Gaussian 두 개를 만든 뒤 정규화한다.

$$
g_0,g_1\overset{\mathrm{iid}}{\sim}\mathcal N(0,1)
+i\mathcal N(0,1),\qquad
|\psi_j\rangle=\frac{(g_0,g_1)^T}
{\sqrt{|g_0|^2+|g_1|^2}}.
$$

복소 Gaussian vector의 방향은 unitary rotation에 불변이므로 정규화된
벡터는 Haar-distributed qubit state가 된다. 이 방식은 별도의 삼각함수 없이
global Haar generator와 동일한 원리를 재사용한다
[[Życzkowski & Sommers 2001, Sec. II.A](https://arxiv.org/html/quant-ph/0012101v3#S2.SS1)].

### 3.3 Product와 global Haar의 차이

Product generator에서 독립적으로 뽑는 것은 `n`개의 2차원 Haar vector이다.
길이 `2^n`의 Gaussian vector 하나를 정규화하는 것이 아니다. 후자는 아래의
global Haar state이며 거의 항상 entangled state를 만든다. 두 상태군은 모두
purity 1이므로, reconstruction 차이를 purity가 아니라 상태 구조와
entanglement 차이로 해석할 수 있다.

---

## 4. Haar-random pure states

### 4.1 Haar measure의 의미

Pure states의 자연스러운 균일 분포는 unitary-invariant Haar measure에서
유도된다. 고정 기준 상태 `|0>`에 Haar-random unitary `U`를 적용한

$$
|\psi\rangle=U|0\rangle
$$

의 분포는 기준 basis와 무관하다. 즉 임의의 고정 unitary `V`에 대해
`|psi>`와 `V|psi>`의 분포가 같다
[[Życzkowski & Sommers 2001](https://arxiv.org/abs/quant-ph/0012101),
[Mezzadri 2007](https://arxiv.org/abs/math-ph/0609050)].

### 4.2 효율적인 생성법: complex Gaussian normalization

전체 `d x d` Haar unitary를 만들 필요는 없다. 다음처럼 길이 `d`의 complex
Gaussian vector 한 개만 정규화하면 Haar-random pure state가 된다.

$$
z_k=x_k+iy_k,\quad x_k,y_k\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),
\qquad
|\psi\rangle=\frac{z}{\sqrt{z^\dagger z}}.
$$

Gaussian의 분산 convention은 정규화에서 사라지므로 실수부·허수부를
`N(0,1)` 또는 `N(0,1/2)`로 택해도 정규화 후 분포는 같다. 이 생성법은
복소 Gaussian 성분의 제곱 크기를 재정규화하면 자연 measure를 얻는다는
결과에 기반한다
[[Życzkowski & Sommers 2001, Eq. (4)--(8) 및 Appendix A](https://arxiv.org/html/quant-ph/0012101v3#S2.SS1)].

Haar unitary 자체가 이후 필요해지는 경우에는 complex Ginibre matrix의 QR
분해 뒤 `R` 대각 성분의 phase를 보정해야 한다. 단순히 QR의 `Q`를 그대로
반환하는 구현은 library convention 때문에 잘못된 분포가 될 수 있다
[[Mezzadri 2007](https://www.ams.org/notices/200705/fea-mezzadri-web.pdf)].
현재 state-vector 생성에는 이 비싼 경로가 필요 없다.

### 4.3 왜 tomography benchmark에 필요한가

Haar-random many-body pure states는 일반적으로 bipartition에 대해 매우 높은
entanglement를 가진다. Page의 결과는 `m <= k`인 bipartition에서 작은
subsystem의 평균 entropy가 거의 최대값 `ln(m)`에 가까움을 보인다
[[Page 1993](https://arxiv.org/abs/gr-qc/9305007)]. 따라서 product state와
Haar state를 비교하면 둘 다 rank 1, purity 1인 상태에서 entanglement와
amplitude structure가 reconstruction에 미치는 영향을 분리할 수 있다.

단, Haar state는 얕은 물리적 회로가 만드는 전형적인 상태와 같다고 가정하면
안 된다. 본 프로젝트에서는 "generic high-entanglement stress test"로
해석하는 것이 적절하다.

---

## 5. Random mixed states: induced Ginibre/Wishart ensemble

### 5.1 mixed state에는 유일한 uniform measure가 없다

Pure-state manifold에는 unitary invariance로 정해지는 자연 Haar measure가
있지만, mixed states의 eigenvalue simplex에는 유일한 자연 measure가 없다.
Hilbert--Schmidt, Bures, induced measure 등 서로 다른 정당한 선택이 서로 다른
purity·eigenvalue 분포를 만든다
[[Życzkowski & Sommers 2001](https://arxiv.org/html/quant-ph/0012101v3#S2)].

따라서 첫 구현은 계산이 단순하고 rank/purity 조절이 명확한 **induced
ensemble**을 기본으로 삼는다.

### 5.2 생성법

`G`를 `d x K` complex Ginibre matrix, 즉 모든 성분의 실수부와 허수부가
독립 Gaussian인 행렬이라 하자. 그러면

$$
W=GG^\dagger,\qquad
\rho=\frac{W}{\operatorname{Tr}W}
$$

는 자동으로 Hermitian, positive semidefinite, trace-one이다. 이는 `d x K`
composite Haar pure state에서 `K`차원 environment를 partial trace한 것과
동일한 induced distribution이다. `K=d`이면 Hilbert--Schmidt measure가 된다
[[Życzkowski & Sommers 2001, Eq. (25)--(32)](https://arxiv.org/html/quant-ph/0012101v3#S3)].

### 5.3 `K`가 제어하는 것

- `K=1`: rank-1 Haar pure state와 동일한 density-matrix ensemble
- `1<K<d`: rank `K`인 low-rank mixed state (확률 1)
- `K>=d`: full-rank mixed state (확률 1)
- `K=d`: Hilbert--Schmidt random state
- `K`가 커질수록 eigenvalues가 `1/d` 부근에 모이고 maximally mixed state에
  가까워진다.

평균 purity는 정확히

$$
\mathbb E_{d,K}[\operatorname{Tr}(\rho^2)]
=\frac{d+K}{dK+1}
$$

이다
[[Życzkowski & Sommers 2001, Eq. (44)](https://arxiv.org/html/quant-ph/0012101v3#S3.SS2)].
중요하게도 이는 **평균값**이며, 개별 sample의 purity를 고정하지 않는다.

`K`/rank는 tomography 난이도의 실제 연구축이다. Low-rank 상태는 일반
full-rank 상태보다 적은 자유도를 가지며, Pauli measurement를 이용한
compressed tomography와 sample-complexity 결과도 rank 의존성을 명시한다
[[Gross et al. 2010](https://arxiv.org/abs/0909.3304),
[Haah et al. 2017](https://arxiv.org/abs/1508.01797)].

### 5.4 권장 sweep

최초 실험에서는 다음을 분리해 기록한다.

- fixed-rank sweep: `K in {1, 2, 4, ..., d}`
- full-rank mixedness sweep: `K in {d, 2d, 4d}`
- 각 `(n, K)`에서 여러 seed를 반복하고 sample purity를 함께 저장

서로 다른 `n`에서 raw `K`만 같게 두면 rank fraction과 평균 purity가 동시에
달라진다. 비교 목적에 따라 `K`, `K/d`, 실제 rank, 실제 purity를 모두
metadata에 남겨야 한다.

---

## 6. 정확한 target purity가 필요할 때

Induced ensemble의 `K`는 purity의 분포만 바꾼다. Challenge의
"varying degrees of purity"를 정확한 독립변수로 실험하려면 별도의
depolarized-pure family가 유용하다.

Haar 또는 product pure state `|psi>`와 `0 <= alpha <= 1`에 대해

$$
\rho_\alpha
=\alpha|\psi\rangle\langle\psi|
+(1-\alpha)\frac{I}{d}
$$

로 두면

$$
\operatorname{Tr}(\rho_\alpha^2)
=\frac{1}{d}+\left(1-\frac{1}{d}\right)\alpha^2.
$$

따라서 원하는 `gamma in [1/d, 1]`에 대해

$$
\alpha
=\sqrt{\frac{\gamma-1/d}{1-1/d}}
$$

를 사용하면 purity가 정확히 `gamma`가 된다. Eigenvalues는

$$
\lambda_1=\alpha+\frac{1-\alpha}{d},\qquad
\lambda_{2,\ldots,d}=\frac{1-\alpha}{d}
$$

이므로 `alpha<1`이면 full rank이다.

이 family는 purity-controlled benchmark이지, fixed-purity surface에서
균일한 random ensemble은 아니다. 또한 실제 장치 noise channel을
시뮬레이션한다고 자동으로 해석해서는 안 된다. 여기서는 state preparation
family일 뿐이며 measurement noise는 기존 규칙대로 finite sampling만 사용한다.

---

## 7. 추가로 유용한 구조적 상태

### 7.1 GHZ state

$$
|\mathrm{GHZ}_n\rangle
=\frac{|0\rangle^{\otimes n}+|1\rangle^{\otimes n}}{\sqrt 2}.
$$

두 개의 basis amplitude만 nonzero이지만 global coherence와 multipartite
entanglement를 가진다. 알려진 Pauli expectation과 density-matrix 원소를
이용하기 쉬워 end-to-end 회귀 상태로 적합하다.

### 7.2 W state

$$
|W_n\rangle
=\frac{1}{\sqrt n}\sum_{j=0}^{n-1}|0\cdots 010\cdots 0\rangle.
$$

한 개 excitation이 모든 qubit에 분산된다. 3-qubit에서 GHZ와 W는 서로 다른
genuine tripartite entanglement class의 대표이며, 한 qubit을 trace out했을 때
보이는 얽힘 성질도 다르다
[[Dür, Vidal & Cirac 2000](https://arxiv.org/abs/quant-ph/0005115)].
따라서 두 상태는 "얽힌 상태"를 하나의 Haar family로만 대표하지 않도록 해
준다.

### 7.3 구현 우선순위

GHZ와 W는 challenge의 최소 필수 항목은 아니다. 먼저 product, Haar,
induced mixed, exact-purity family를 구현하고, GHZ/W는 convention regression과
poster의 해석 가능한 예시가 필요할 때 추가하는 것이 좋다.

---

## 8. 제안하는 구현 계약

구체적 함수명은 구현 시 조정할 수 있지만, 의미는 아래처럼 분리한다.

```text
random_product_pure_state(like, n, seed=None) -> rho (and optionally ket)
random_haar_pure_state(like, n, seed=None)    -> rho (and optionally ket)
random_induced_state(like, n, K, seed=None)  -> rho
random_state_with_purity(like, n, gamma, seed=None, base="haar") -> rho
ghz_state(like, n) -> rho
w_state(like, n)   -> rho
```

### 8.1 Array API와 device

State generator에는 기존 `rho` 입력이 없으므로 backend를 추론할
`like`/prototype array를 첫 인자로 받는 방식이 프로젝트 규칙에 가장 잘
맞는다.

```text
xp = array_namespace(like)
device = getattr(like, "device", None)
```

이후 모든 array 생성과 선형대수는 `xp`에서 수행한다.

- core module에서 NumPy를 직접 import하지 않는다.
- `xp.asarray(..., dtype=xp.complex128, device=device)`로 상수를 옮긴다.
- outer product는 `psi[:, None] * psi.conj()[None, :]`처럼 함수형으로 만든다.
- `arr[i] = value` 같은 in-place update는 사용하지 않는다.
- tensor product는 qubit 0부터 `xp.kron`한다.
- JAX는 생성 전에 `jax_enable_x64 = True`가 필요하다.

### 8.2 난수와 재현성

난수의 유일한 원천은 하나의 `random.Random(seed)` instance여야 한다.

1. 실수부와 허수부 Gaussian draw를 Python list에 고정 순서로 만든다.
2. list를 한 번에 해당 backend/device array로 변환한다.
3. backend RNG (`np.random`, `jax.random`, `torch.rand`, `cupy.random`)는
   호출하지 않는다.

같은 seed의 **난수 draw와 상태 family는 모든 backend에서 동일**해야 한다.
Dense matrix multiplication/reduction의 마지막 bit는 backend별 연산 순서로
달라질 수 있으므로, backend state 비교는 `complex128`의 엄격한 tolerance를
사용한 수치 동등성으로 정의한다. Measurement dataset의 fixed-seed 결과도
기존 테스트와 연결해 확인한다.

### 8.3 반환 표현

Measurement simulator가 density matrix를 입력받으므로 public generator의
기본 반환값은 `(d, d)` density matrix가 가장 단순하다. Pure-state overlap과
memory-efficient 후속 연구를 위해 ket이 필요하다면 다음 중 하나를 명시적으로
선택한다.

- `return_ket=True`로 `(ket, rho)` 반환
- frozen result dataclass에 `rho`, optional `ket`, `family`, `seed`, parameters 저장

반환형을 정하기 전에는 tuple 위치에 의미를 암묵적으로 부여하지 않는 것이
좋다. 어떤 방식을 택하든 `rho`, family name, `K`/target purity와 seed가
실험 기록에서 분리되지 않아야 한다.

---

## 9. 복잡도와 범위 제한

| 생성기 | 주 난수 개수 | 임시 표현 | density matrix까지 만들 때 메모리 |
|---|---:|---:|---:|
| Product pure | `4n` real Gaussian | local kets + global ket | `O(d^2)` |
| Haar pure | `2d` real Gaussian | global ket | `O(d^2)` |
| Induced `(d,K)` | `2dK` real Gaussian | `G`와 `rho` | `O(dK+d^2)` |

Density matrix 자체가 `4^n` complex entries를 가지므로 `n=20`까지 dense
state와 full QST를 그대로 확장하는 것은 현실적이지 않다. 큰 `n` benchmark는
ket/low-rank factor를 유지하거나 reconstruction protocol을 줄이는 별도 설계가
필요하다. State generator가 빠르다는 사실이 dense tomography 전체의
scalability를 뜻하지 않는다.

---

## 10. 구현 후 필수 검증 계획

AI 생성 코드와 수식은 아래 검증을 통과해도 사람이 결과를 검토하기 전에는
verified로 표시하지 않는다.

### 10.1 결정적 물리 검증

모든 생성 state에 대해 다음을 검사한다.

- shape `(2**n, 2**n)`와 dtype `complex128`
- `rho == rho.conj().T`
- `Tr(rho) == 1`
- eigenvalues `>= -tolerance`
- pure family의 `Tr(rho^2) == 1`
- target-purity family의 `Tr(rho^2) == gamma`
- induced state의 numerical rank가 `min(d,K)`와 일치

### 10.2 구조와 convention 회귀

- `n=1` product와 Haar 생성기의 동일 분포적 정의
- `|q0 q1 ...>`의 nonzero amplitude index가 MSB 규칙과 일치
- GHZ의 nonzero amplitudes가 index `0`, `d-1`
- W의 nonzero amplitudes가 `2**(n-1-j)` 위치
- generated `|+y>`를 measurement module에 넣었을 때 `<Y>=+1`

### 10.3 seed/backend 검증

- 같은 backend, 같은 seed: 동일 결과
- 다른 seed: 일반적으로 다른 결과
- NumPy/JAX 및 설치된 CuPy/PyTorch에서 같은 seed의 ket/rho가 tolerance 내 일치
- 생성 state를 기존 measurement generator에 전달했을 때 fixed-seed outcome과
  count가 backend 간 일치

### 10.4 분포 검증은 별도 통계 테스트로

Flaky한 Monte Carlo 검사를 빠른 unit test에 넣기보다 고정 seed의 분석
script로 다음을 확인하고 결과를 문서화한다.

- single-qubit Haar Bloch vector의 평균이 `(0,0,0)`에 가까움
- 각 Bloch component의 second moment가 `1/3`에 가까움
- Haar state의 평균 basis probability가 `1/d`에 가까움
- induced ensemble의 sample mean purity가 `(d+K)/(dK+1)`에 수렴

표본 수, seed set, 오차막대와 허용 기준을 함께 기록해야 한다.

---

## 11. 흔한 오류와 해석상 주의

1. **각 amplitude를 uniform distribution으로 뽑고 정규화**하면 일반적으로
   Haar state가 아니다. Isotropic complex Gaussian을 사용한다.
2. **Bloch polar angle `theta`를 uniform하게 뽑으면** sphere-uniform이 아니다.
   `cos(theta)`를 uniform하게 뽑거나 complex Gaussian을 정규화한다.
3. **`G`를 Hermitian으로 먼저 만들지 않는다.** Ginibre `G`는 일반 rectangular
   complex matrix이고, positive matrix는 `G G^dagger`에서 생긴다.
4. **`K`는 exact purity가 아니다.** 평균 purity와 rank를 조절하는 ensemble
   parameter이다.
5. **Depolarized pure state는 induced ensemble이 아니다.** 같은 purity라도
   eigenvalue spectrum 분포가 다르므로 결과를 합쳐 보고하지 않는다.
6. **Global phase 차이를 state-vector failure로 판단하지 않는다.** Density
   matrix와 pure-state overlap은 global phase에 불변이다.
7. **Random state 생성과 measurement noise를 혼동하지 않는다.** 생성된 `rho`가
   target이며, measurement module은 그 target의 정확한 Born distribution에서
   finite shots만 추출한다.
8. **Haar가 모든 물리적 상태를 대표한다고 해석하지 않는다.** Haar는 명확한
   수학적 ensemble이자 stress test이고, GHZ/W/product는 구조적 비교군이다.

---

## 12. 실험 설계 권고

Challenge의 "state class별 reconstruction quality" 질문에는 다음 factorial
design이 해석하기 쉽다.

1. `n`과 shots-per-setting을 고정한다.
2. pure comparison에서 product와 Haar를 비교한다. 두 family 모두 purity 1이다.
3. rank comparison에서 induced ensemble의 `K`를 바꾼다.
4. purity comparison에서 exact-purity family의 `gamma`를 바꾼다.
5. 각 cell에서 여러 state seed와 measurement seed를 독립적으로 반복한다.
6. fidelity/overlap의 평균뿐 아니라 분산 또는 confidence interval을 보고한다.

State seed와 measurement seed를 분리해야 "어떤 target이 뽑혔는가"와
"그 target에서 어떤 finite-shot dataset이 뽑혔는가"를 재현하고, state-to-state
variation과 shot noise를 구분할 수 있다.

---

## 13. 주요 문헌

1. K. Życzkowski and H.-J. Sommers, **Induced measures in the space of mixed
   quantum states**, *J. Phys. A* 34, 7111 (2001).
   [arXiv](https://arxiv.org/abs/quant-ph/0012101) ·
   [DOI](https://doi.org/10.1088/0305-4470/34/35/335)
2. F. Mezzadri, **How to generate random matrices from the classical compact
   groups**, *Notices of the AMS* 54, 592 (2007).
   [arXiv](https://arxiv.org/abs/math-ph/0609050) ·
   [AMS PDF](https://www.ams.org/notices/200705/fea-mezzadri-web.pdf)
3. D. N. Page, **Average Entropy of a Subsystem**, *Phys. Rev. Lett.* 71,
   1291 (1993). [arXiv](https://arxiv.org/abs/gr-qc/9305007) ·
   [DOI](https://doi.org/10.1103/PhysRevLett.71.1291)
4. D. Gross, Y.-K. Liu, S. T. Flammia, S. Becker, and J. Eisert,
   **Quantum state tomography via compressed sensing**, *Phys. Rev. Lett.*
   105, 150401 (2010). [arXiv](https://arxiv.org/abs/0909.3304) ·
   [DOI](https://doi.org/10.1103/PhysRevLett.105.150401)
5. J. Haah, A. W. Harrow, Z. Ji, X. Wu, and N. Yu,
   **Sample-optimal tomography of quantum states**, *IEEE Trans. Inf. Theory*
   63, 5628 (2017). [arXiv](https://arxiv.org/abs/1508.01797) ·
   [DOI](https://doi.org/10.1109/TIT.2017.2719044)
6. W. Dür, G. Vidal, and J. I. Cirac, **Three qubits can be entangled in two
   inequivalent ways**, *Phys. Rev. A* 62, 062314 (2000).
   [arXiv](https://arxiv.org/abs/quant-ph/0005115) ·
   [DOI](https://doi.org/10.1103/PhysRevA.62.062314)

---

## 14. 검증 상태 기록

- 문헌 링크 및 bibliographic metadata: AI가 원 논문/arXiv 페이지를 조사해 작성
- 수식 전개: AI가 작성했으며 독립 검토 전
- 구현: 아직 수행하지 않음
- 수치/통계 검증: 아직 수행하지 않음
- 현재 상태: **UNVERIFIED — independent review required**
