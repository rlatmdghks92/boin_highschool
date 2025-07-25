import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import sigma  # 슈테판-볼츠만 상수
import platform

# 운영체제에 따라 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'    
else:
    plt.rcParams['font.family'] = 'NanumGothic'     
plt.rcParams['axes.unicode_minus'] = False  

# 상수 정의
epsilon = 0.95              # 방사율
alpha_grass = 0.25          # 인조잔디 알베도
alpha_sand = 0.4            # 모래 알베도
S = 1000                    # 태양 복사 세기 (W/m²)
rho = 1.2                   # 공기 밀도 (kg/m³)
cp = 1005                   # 공기 비열 (J/kg·K)
k = 0.2                     # 복사 감쇠 계수 (1/m)
k_prime = 0.05              # 흡수 감쇠 계수 (1/m)
T_infty = 36                # 배경 공기 온도 (°C)

# 표면 온도 (섭씨, 켈빈)
T_sC_grass = 70.19
T_sC_sand = 52.11
T_sK_grass = T_sC_grass + 273.15
T_sK_sand = T_sC_sand + 273.15

# 고도 범위 (0 ~ 5m, 0.5m 간격)
h_vals = np.arange(0, 5.01, 0.5)

# 고도별 기온 계산 함수
def compute_T_no_convection(T_s_K, alpha):
    radiation_term = (epsilon * sigma * T_s_K ** 4) / (rho * cp * k) * np.exp(-k * h_vals)
    solar_absorption_term = (alpha * S) / (rho * cp * k_prime) * np.exp(-k_prime * h_vals)
    T_h = T_infty + radiation_term + solar_absorption_term
    return T_h

# 인조잔디와 모래 각각 계산
T_grass_profile = compute_T_no_convection(T_sK_grass, alpha_grass)
T_sand_profile = compute_T_no_convection(T_sK_sand, alpha_sand)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(h_vals, T_grass_profile, label='인조 잔디', color='green')
plt.plot(h_vals, T_sand_profile, label='모래', color='saddlebrown')

# 라벨 및 제목
plt.xlabel('고도 (m)')
plt.ylabel('기온 (°C)')
plt.title('고도에 따른 기온 (복사만 고려)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
