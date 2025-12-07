import numpy as np
import matplotlib.pyplot as plt


print("1. Квадратная волна")
print("2. |cos(t)| (чётная)")
print("3. sin(t) (нечётная)")
print("4. sin(t) + 0.5cos(2t) (произвольная)")
choice = input("Введите номер (1-4): ")


if choice == "1":
    # Квадратная волна
    a = 1.0
    b = 2.0
    t0 = 0.0
    t1 = 1.0
    t2 = 2.0
    T = t2 - t0 
    
    def f(t):
        t_mod = (t - t0) % T
        return a if t_mod < (t1 - t0) else b
    
    func_name = "Квадратная волна"
    xlim_factor = 3  
    
elif choice == "2":
    # |cos(t)|
    T = np.pi
    
    def f(t):
        return np.abs(np.cos(t))
    
    func_name = "|cos(t)|"
    xlim_factor = 3
    
elif choice == "3":
    # sin(t)
    T = 2 * np.pi
    
    def f(t):
        return np.sin(t)
    
    func_name = "sin(t)"
    xlim_factor = 2
    
elif choice == "4":
    # sin(t) + 0.5cos(2t)
    T = 2 * np.pi
    
    def f(t):
        return np.sin(t) + 0.5 * np.cos(2*t)
    
    func_name = "sin(t) + 0.5cos(2t)"
    xlim_factor = 2
    
else:
    print("Неверный выбор!")
    exit()


def fourier_coefficients_real(f, T, N):
    #Вычисляет коэффициенты a_n, b_n для вещественного ряда Фурье.
    omega0 = 2 * np.pi / T
    num_points = 10000
    t_vals = np.linspace(0, T, num_points)
    f_vals = np.array([f(t) for t in t_vals])
    
    a_coeffs = []
    b_coeffs = []
    
    # a0
    a0 = (2 / T) * np.trapezoid(f_vals, t_vals)
    a_coeffs.append(a0)
    
    # a_n, b_n для n = 1..N
    for n in range(1, N + 1):
        cos_vals = f_vals * np.cos(n * omega0 * t_vals)
        sin_vals = f_vals * np.sin(n * omega0 * t_vals)
        
        an = (2 / T) * np.trapezoid(cos_vals, t_vals)
        bn = (2 / T) * np.trapezoid(sin_vals, t_vals)
        
        a_coeffs.append(an)
        b_coeffs.append(bn)
    
    return a_coeffs, b_coeffs

def fourier_coefficients_complex(f, T, N):
    #Вычисляет коэффициенты c_n для комплексного ряда Фурье.
    omega0 = 2 * np.pi / T
    num_points = 10000
    t_vals = np.linspace(0, T, num_points)
    f_vals = np.array([f(t) for t in t_vals])
    
    c_coeffs = []
    for n in range(-N, N + 1):
        exp_vals = f_vals * np.exp(-1j * n * omega0 * t_vals)
        cn = (1 / T) * np.trapezoid(exp_vals, t_vals)
        c_coeffs.append(cn)
    
    return c_coeffs

def compute_FN(t, a_coeffs, b_coeffs, T):
    #Вычисляет частичную сумму вещественного ряда Фурье F_N(t).
    omega0 = 2 * np.pi / T
    result = a_coeffs[0] / 2  # a0/2
    
    for n in range(1, len(a_coeffs)):
        result += a_coeffs[n] * np.cos(n * omega0 * t)
    
    for n in range(1, len(b_coeffs) + 1):
        result += b_coeffs[n-1] * np.sin(n * omega0 * t)
    
    return result

def compute_GN(t, c_coeffs, T, N):
    #Вычисляет частичную сумму комплексного ряда Фурье G_N(t).
    omega0 = 2 * np.pi / T
    result = 0j
    
    for idx, cn in enumerate(c_coeffs):
        n = idx - N  
        result += cn * np.exp(1j * n * omega0 * t)
    
    return result.real


print("\n" + "=" * 60)
print(f"функция: {func_name}, период T = {T}")
print("=" * 60)


N_values = [1, 2, 5, 10, 300]

max_N = max(N_values)

# Вычисляем коэффициенты для максимального N
a_coeffs_full, b_coeffs_full = fourier_coefficients_real(f, T, max_N)
c_coeffs_full = fourier_coefficients_complex(f, T, max_N)



print("Коэффициенты a_n (n = 0..2):")
for i in range(min(3, len(a_coeffs_full))):
    print(f"a_{i} = {a_coeffs_full[i]:.6f}")

print("\nКоэффициенты b_n (n = 1..2):")
for i in range(1, min(3, len(b_coeffs_full) + 1)):
    print(f"b_{i} = {b_coeffs_full[i-1]:.6f}")

print("\nКоэффициенты c_n (n = -2..2):")
for idx in range(min(5, len(c_coeffs_full))):
    n = idx - 2
    cn = c_coeffs_full[idx]
    sign = '+' if cn.imag >= 0 else ''
    print(f"c_{n} = {cn.real:.6f} {sign}{abs(cn.imag):.6f}j")



t_plot = np.linspace(-T, xlim_factor*T, 1000)
f_original = np.array([f(t) for t in t_plot])

print(f"\nПостроение графиков для N = {N_values}...")

# Создаём графики для каждого N
for N in N_values:
    plt.figure(figsize=(12, 6))
    

    if N <= max_N:
        a_coeffs_N = [a_coeffs_full[0]] + a_coeffs_full[1:N+1]
        b_coeffs_N = b_coeffs_full[:N]
        c_coeffs_N = c_coeffs_full[max_N-N:max_N+N+1]
    else:
        a_coeffs_N = a_coeffs_full
        b_coeffs_N = b_coeffs_full
        c_coeffs_N = c_coeffs_full
    
    FN_values = np.array([compute_FN(t, a_coeffs_N, b_coeffs_N, T) for t in t_plot])
    GN_values = np.array([compute_GN(t, c_coeffs_N, T, N) for t in t_plot])
    
    plt.plot(t_plot, f_original, 'k-', linewidth=3, alpha=0.3, label='Исходная f(t)')
    plt.plot(t_plot, FN_values, 'b-', linewidth=2, label=f'F_N(t), N={N}')
    plt.plot(t_plot, GN_values, 'r--', linewidth=2, label=f'G_N(t), N={N}')
    
    plt.title(f'{func_name}: частичные суммы Фурье при N={N}')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-T, xlim_factor*T])
    plt.tight_layout()
    plt.show()

#Проверка равенства Парсеваля

# Используем максимальное N
N_max = max_N

# Вычисляем "энергию" исходного сигнала на одном периоде
t_period = np.linspace(0, T, 10000)
f_period = np.array([f(t) for t in t_period])
energy_original = (1/T) * np.trapezoid(f_period**2, t_period)

# Вычисляем через коэффициенты Фурье
energy_FN = (a_coeffs_full[0]**2) / 4
for n in range(1, min(N_max + 1, len(a_coeffs_full))):
    energy_FN += 0.5 * (a_coeffs_full[n]**2 + b_coeffs_full[n-1]**2)

energy_GN = 0
for idx, cn in enumerate(c_coeffs_full):
    energy_GN += abs(cn)**2

print(f"Средняя мощность исходного сигнала: {energy_original:.8f}")
print(f"Сумма квадратов коэффициентов F_N (N={N_max}): {energy_FN:.8f}")
print(f"Сумма квадратов коэффициентов G_N (N={N_max}): {energy_GN:.8f}")
print(f"Разница F_N: {abs(energy_original - energy_FN):.10f}")
print(f"Разница G_N: {abs(energy_original - energy_GN):.10f}")


# Проверка
tolerance = 1e-3
if abs(energy_original - energy_FN) < tolerance and abs(energy_original - energy_GN) < tolerance:
    print(f"\n Равенство Парсеваля выполняется")
else:
    print(f"\n Для большей точности нужно увеличить число гармоник N")
