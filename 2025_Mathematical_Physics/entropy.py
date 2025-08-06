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

# 초기 상태 설정
theta = np.pi / 4
rho00 = np.cos(theta)**2
rho11 = np.sin(theta)**2
rho01_0 = np.cos(theta) * np.sin(theta)

# 파라미터
Gamma = 1.0
t_vals = np.linspace(0, 5, 100)
entropy_vals = []
eig1_vals = []
eig2_vals = []

# 시간에 따른 밀도 행렬 구성 → 고윳값, 엔트로피 계산
for t in t_vals:
    rho01_t = rho01_0 * np.exp(-Gamma * t)
    rho = np.array([[rho00, rho01_t],
                    [rho01_t, rho11]])
    eigenvalues = np.linalg.eigvalsh(rho)
    entropy = -np.sum([l * np.log2(l) for l in eigenvalues if l > 0])
    entropy_vals.append(entropy)
    eig1_vals.append(eigenvalues[0])
    eig2_vals.append(eigenvalues[1])

# 시각화
plt.figure(figsize=(12, 4))

# (1) 고윳값 변화
plt.subplot(1, 2, 1)
plt.plot(t_vals, eig1_vals, label="고윳값 1")
plt.plot(t_vals, eig2_vals, label="고윳값 2")
plt.xlabel("시간 t")
plt.ylabel("고윳값")
plt.title("시간에 따른 밀도 행렬 고윳값 변화")
plt.legend()
plt.grid(True)

# (2) 폰 노이만 엔트로피
plt.subplot(1, 2, 2)
plt.plot(t_vals, entropy_vals, color='green')
plt.xlabel("시간 t")
plt.ylabel("엔트로피")
plt.title("폰 노이만 엔트로피의 시간 변화")
plt.grid(True)

plt.tight_layout()
plt.show()
