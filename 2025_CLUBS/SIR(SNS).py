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
I0 = 1             # (초기) 정보를 접하고 공유하는 사람 수
R0 = 0             # (초기) 정보를 더 이상 공유하지 않는 사람 수
S0 = N - I0 - R0   # (초기) 정보를 접하지 않은 사람 수

beta = 2.5         # 정보 전파율
gamma = 0.5        # 정보 소멸률

# 시간 범위 (일 단위)
t = np.linspace(0, 14, 14)

# 2. SIR 미분방정식 정의
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
plt.plot(t, S, 'b', label='정보를 접하지 않은 사람 S(t)')
plt.plot(t, I, 'r', label='정보를 접하고 공유하는 사람 I(t)')
plt.plot(t, R, 'g', label='정보를 더 이상 공유하지 않는 사람 R(t)')
plt.xlabel('시간 (일)')
plt.ylabel('인구 수')
plt.title(f'정보 확산 모형 시뮬레이션(SIR) (β={beta}, γ={gamma})')
plt.legend()
plt.grid()
plt.show()
