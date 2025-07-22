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

beta = 0.25         # 감염률, 코로나 유행 초기는 0.25, 유행 이후는 1.8로 설정
gamma = 0.1        # 회복률, 코로나 유행 초기는 0.1, 유행 이후는 0.2로 설정
xi = 0.008          # 면역 상실률 (면역 지속 100일 가정)

# 시간 범위 (일 단위)
t = np.linspace(0, 250, 250)

# 2. SIRS 미분방정식 정의
def deriv(y, t, N, beta, gamma, xi):
    S, I, R = y
    dSdt = -beta * S * I / N + xi * R
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I - xi * R
    return dSdt, dIdt, dRdt

# 초기 조건
y0 = S0, I0, R0

# 미분방정식 풀기
ret = odeint(deriv, y0, t, args=(N, beta, gamma, xi))
S, I, R = ret.T

# 3. 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='감염 가능자 S(t)')
plt.plot(t, I, 'r', label='감염자 I(t)')
plt.plot(t, R, 'g', label='회복자 R(t)')
plt.xlabel('시간 (일)')
plt.ylabel('인구 수')
plt.title(f'SIRS 모형 시뮬레이션 (β={beta}, γ={gamma}, ξ={xi})')
plt.legend()
plt.grid()
plt.show()
