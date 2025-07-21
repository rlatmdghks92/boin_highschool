import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ipywidgets import interact, FloatSlider
import platform
import warnings
warnings.filterwarnings("ignore")

# ────────────── 한글 폰트 설정 ──────────────
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False

# ────────────── SIR 시뮬레이션 함수 ──────────────
def simulate_sir(beta=0.3, gamma=0.1):
    N = 1000
    I0 = 1
    R0 = 0
    S0 = N - I0 - R0
    t = np.linspace(0, 160, 160)

    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    y0 = S0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='감염 가능자 S(t)')
    plt.plot(t, I, label='감염자 I(t)')
    plt.plot(t, R, label='회복자 R(t)')
    plt.xlabel('시간 (일)')
    plt.ylabel('인구 수')
    plt.title(f'SIR 모형 시뮬레이션 (β={beta:.2f}, γ={gamma:.2f})')
    plt.legend()
    plt.grid()
    plt.show()

# ────────────── 슬라이더 인터페이스 ──────────────
interact(simulate_sir,
         beta=FloatSlider(value=0.3, min=0.05, max=1.0, step=0.01, description='감염률 β'),
         gamma=FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01, description='회복률 γ'))
