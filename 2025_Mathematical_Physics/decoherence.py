import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

# 시뮬레이션 데이터
theta = np.pi / 4
rho01 = np.cos(theta) * np.sin(theta)
Gamma = 1.0
t_vals = np.linspace(0, 5, 100)
rho01_vals = rho01 * np.exp(-Gamma * t_vals)

# 시각화
plt.figure(figsize=(7, 4))
plt.plot(t_vals, rho01_vals, label=r"$\rho_{01}(t)$")
plt.xlabel("시간 t")
plt.ylabel("간섭항(비대각 성분)")
plt.title("디코히런스에 의한 간섭항의 감소")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
