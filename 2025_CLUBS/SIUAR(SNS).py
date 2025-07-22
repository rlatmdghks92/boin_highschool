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
N = 1000            # 총 인구 수
I0 = 1              # (초기) 거짓 정보를 접하고 공유하는 사람 수
S0 = N - I0         # (초기) 정보를 접하지 않은 사람 수
U0 = 0              # (초기) 거짓 정보를 더 이상 공유하지 않는 사람 수
A0 = 1              # (초기) 사실 정보를 접하고 공유하는 사람 수
R0 = 0              # (초기) 사실 정보를 더 이상 공유하지 않는 사람 수

beta = 2.5     # 정보 전파율
gamma = 0.5    # 정보 소멸률
t_A = 7        # A 등장 시점

# 시간 범위 (일 단위)
t = np.linspace(0, 21, 21)

# 모델 정의
def model(y, t):
    S, I, U, A, R = y

    if t < t_A:
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dUdt = gamma * I
        dAdt = 0
        dRdt = 0
    else:
        dSdt = -beta * S * I / N - beta * S * A / N
        dIdt = beta * S * I / N - gamma * I - beta * I * A / N
        dUdt = gamma * I - beta * U * A / N
        dAdt = beta * (S * A + I * A + U * A) / N - gamma * A
        dRdt = gamma * A

    return [dSdt, dIdt, dUdt, dAdt, dRdt]

# 초기 조건
y0 = [S0, I0, U0, A0, R0]

# 미분 방정식 풀기
result = odeint(model, y0, t)
S, I, U, A, R = result.T

# 3. 그래프 그리기 (시각화)
plt.figure(figsize=(12, 6))
plt.plot(t, S, label='S: 정보를 접하지 않은 사람', color='gray', linewidth=2)
plt.plot(t, I, label='I: 거짓 정보를 접하고 공유하는 사람', color='red', linewidth=2)
plt.plot(t, U, label='U: 거짓 정보를 더 이상 공유하지 않는 사람', color='orange', linewidth=2)
plt.plot(t, A, label='A: 사실 정보를 접하고 공유하는 사람', color='blue', linewidth=2)
plt.plot(t, R, label='R: 사실 정보를 더 이상 공유하지 않는 사람', color='green', linewidth=2)
plt.axvline(x=t_A, color='black', linestyle='--', linewidth=1.5, label='A 등장 시점')

plt.xlabel('시간 (일)')
plt.ylabel('인구 수')
plt.title('정보 확산 모형 시뮬레이션(SIUAR) (전파율 β = 2.5 소멸률 γ = 0.5)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
