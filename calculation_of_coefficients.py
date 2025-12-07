import numpy as np


print("Выберите функцию:")
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
    
    def function(t):
        t_mod = (t - t0) % T
        return a if t_mod < (t1 - t0) else b
    
    func_name = "Квадратная волна"
    
elif choice == "2":
    # |cos(t)|
    T = np.pi
    
    def function(t):
        return np.abs(np.cos(t))
    
    func_name = "|cos(t)|"
    
elif choice == "3":
    # sin(t)
    T = 2 * np.pi
    
    def function(t):
        return np.sin(t)
    
    func_name = "sin(t)"
    
elif choice == "4":
    # sin(t) + 0.5cos(2t)
    T = 2 * np.pi
    
    def function(t):
        return np.sin(t) + 0.5 * np.cos(2*t)
    
    func_name = "sin(t) + 0.5cos(2t)"
    
else:
    print("Неверный выбор!")
    exit()


def fourier_coefficients_real(f, T, N):
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

N = 2
a_coeffs, b_coeffs = fourier_coefficients_real(function, T, N)
c_coeffs = fourier_coefficients_complex(function, T, N)



print(f"Функция: {func_name}")
print(f"Период T = {T}")

print("\nКоэффициенты a_n (n = 0..2):")

for i, an in enumerate(a_coeffs):
    print(f"a_{i} = {an:.6f}")

print("\nКоэффициенты b_n (n = 1..2):")
for i, bn in enumerate(b_coeffs, start=1):
    print(f"b_{i} = {bn:.6f}")

print("\nКоэффициенты c_n (n = -2..2):")
for idx, cn in enumerate(c_coeffs):
    n = idx - N
    sign = '+' if cn.imag >= 0 else '-'
    print(f"c_{n} = {cn.real:.6f} {sign} {abs(cn.imag):.6f}j")