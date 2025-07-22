import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import platform

# 운영체제에 따라 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 1. 파라미터 설정
N = 1000           # 총 인구 수
I0 = 1             # 초기 감염자 수
R0 = 0             # 초기 회복자 수
S0 = N - I0 - R0   # 초기 감염 가능자 수

beta = 1.8        # 감염률, 코로나 유행 초기는 0.25, 유행 이후는 1.8로 설정
gamma = 0.2        # 회복률, 코로나 유행 초기는 0.1, 유행 이후는 0.2로 설정

# 시간 범위 (일 단위)
t = np.linspace(0, 160, 160)

# 2. SIR 미분방정식 정의
# ───────────────────────────────────────
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# 초기 조건
y0 = S0, I0, R0

# 미분방정식 풀기
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# 3. 그래프 그리기 (시각화)
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='감염 가능자 S(t)')
plt.plot(t, I, 'r', label='감염자 I(t)')
plt.plot(t, R, 'g', label='회복자 R(t)')
plt.xlabel('시간 (일)')
plt.ylabel('인구 수')
plt.title(f'SIR 모형 시뮬레이션 (β={beta}, γ={gamma})')
plt.legend()
plt.grid()
plt.show()
