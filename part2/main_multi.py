## Практическое задание № 5. Линейная регрессия (со множеством переменных)

# Инициализация
import numpy as np
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from part1.computeCost import computeCost
from part1.gradientDescent import gradientDescent
from normalEqn import normalEqn

# =================== Часть 1. Загрузка данных ===================

print('Часть 1. Загрузка данных')

# Загрузка данных и формирование матрицы объекты-признаки X и вектора меток y
data = np.loadtxt('data2.txt', delimiter =',')
m = data.shape[0]
X = np.array(data[:, 0:2])
y = np.array(data[:, 2:3])

# ================ Часть 2. Нормализация признаков ===============

print('Часть 2. Нормализация признаков')

# Выполнение нормализации признаков
X, mu, sigma = featureNormalize(X)
X = np.concatenate((np.ones((m, 1)), X), axis = 1)

# ================== Часть 3. Градиентный спуск ==================

print('Часть 3. Градиентный спуск')

# Задание начальных значений параметров модели
theta = np.zeros([3, 1])

# Вычисление значения стоимостной функции для начального theta
J = computeCost(X, y, theta)

# Задание параметров градиентного спуска
iterations = 400
alpha = 0.01

# Выполнение градиентного спуска
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print('Найденные параметры модели: ')
print(theta)

# Визуализация процесса сходимости
plt.figure()
plt.plot(np.arange(len(J_history)) + 1, J_history, '-b', linewidth = 2)
plt.xlabel('Число итераций')
plt.ylabel('Значение стоимостной функции')
plt.grid()
plt.show()

# Предсказание стоимости жилья площадью 1650 квадратных футов и 3 комнат
predict = np.dot(np.array([[1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]]]), theta)
print('Для площади 1650 квадратных футов и 3 комнат стоимость жилья = ${:.7f}'.format(predict[0, 0]))

# =================== Исследование влияния числа итераций при фиксированном α ===================
print('Исследование влияния числа итераций при α=x')

alpha_fixed = 0.01
iterations_list = [10, 50, 100, 200]
colors = ['r', 'g', 'b', 'm']
linestyles = ['-', '--', '-.', ':']
plt.figure(figsize=(10, 6))

for i, iter_count in enumerate(iterations_list):
    theta_test = np.zeros([3, 1])
    theta_test, J_hist = gradientDescent(X, y, theta_test, alpha_fixed, iter_count)
    plt.plot(np.arange(1, len(J_hist) + 1), J_hist, color=colors[i], linestyle=linestyles[i], linewidth=2, label=f'{iter_count} итераций')

plt.xlabel('Число итераций')
plt.ylabel('Значение стоимостной функции J(θ)')
plt.title(f'Влияние числа итераций при α={alpha_fixed}')
plt.legend()
plt.grid(True)
plt.show()

# ================= Часть 4. Нормальные уравнения ================

print('Часть 4. Нормальные уравнения')

# Загрузка данных и формирование матрицы объекты-признаки X и вектора меток y
data = np.loadtxt('data2.txt', delimiter =',')
m = data.shape[0]
X = np.array(data[:, 0:2]); X = np.concatenate((np.ones((m, 1)), X), axis = 1)
y = np.array(data[:, 2:3])

# Вычисление параметров модели с использованием нормальных уравнений
theta = normalEqn(X, y)
print('Найденные параметры модели: ')
print(theta)

# Предсказание стоимости жилья площадью 1650 квадратных футов и 3 комнат
predict = np.dot(np.array([[1, 1650, 3]]), theta)
print('Для площади 1650 квадратных футов и 3 комнат стоимость жилья = ${:.7f}'.format(predict[0, 0]))