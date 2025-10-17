import numpy as np
from part1.computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели theta, используя матрицу объекты-признаки X, 
        вектор меток y, параметр сходимости alpha и число итераций 
        алгоритма num_iters
    """
    
    J_history = []
    m = y.shape[0]
    
    for i in range(num_iters):

        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить градиентный спуск для num_iters итераций 
        # с целью вычисления вектора параметров theta, минимизирующего 
        # стоимостную функцию

        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = (1 / m) * np.dot(X.T, errors)
        theta = theta - alpha * gradient

        # ============================================================
        
        J_history.append(computeCost(X, y, theta)) # сохранение значений стоимостной функции
                                                   # на каждой итерации
    
    return theta, J_history