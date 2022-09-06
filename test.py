import numpy as np
import matplotlib.pyplot as plt



alpha = 0.5; eps =0.01


#  Исходная функция

def f(x, y):
   #return (4.3 * pow(x, 2)) - (6.4 * x * y) + (3.1 * pow(y, 2)) + (8.9 * x) - 12.89 * y + 12.6
   return ((np.sin(x - 1) + y - 0.1) ** 2) + ((x - np.sin(y + 1) - 0.8) ** 2)


X_m = [];  Y_m = []; F_m = []


def cond(x, y, x0, y0, alpha):

    if f(x, y) > f(x0, y0):

        alpha = alpha / 2

        x = x0 + g_1 * alpha;
        y = y0 + g_2 * alpha

        x, y, x0, y0, alpha = cond(x, y, x0, y0, -alpha)

    return x, y, x0, y0, alpha

x0 = 0;  y0 = 0



d_x = 2 * x0 + 2 * (y0 + np.sin(x0 - 1) - 0.1) * np.cos(x0 - 1) - 2 * np.sin(y0 + 1) - 1.6
d_y = 2 * y0 - 2 * (x0 - np.sin(y0 + 1) - 0.8) * np.cos(y0 + 1) + 2 * np.sin(x0 - 1) - 0.2


for i in range(100):

    if d_x == 0 or d_y ==0 : break

    g_1 = d_x * (pow(d_x, 2) + pow(d_y, 2)) ** (-0.5)
    g_2 = d_y * (pow(d_x, 2) + pow(d_y, 2)) ** (-0.5)

    x = x0 - g_1 * alpha;
    y = y0 - g_2 * alpha

    if f(x, y) > f(x0, y0):

        x, y, x0, y0, alpha = cond(x, y, x0, y0, alpha)


        F_m.append(f(x, y))

        X_m.append(x)
        Y_m.append(y)

        print('x = ', x, '\ty = ', y, '\tf(x, y) = ', f(x, y), '\t| № итерации: ', i, '\nШаг h = ',alpha, '\n')

        print('--------------------------------------------')

        d_x = 2 * x0 + 2 * (y0 + np.sin(x0 - 1) - 0.1) * np.cos(x0 - 1) - 2 * np.sin(y0 + 1) - 1.6
        d_y = 2 * y0 - 2 * (x0 - np.sin(y0 + 1) - 0.8) * np.cos(y0 + 1) + 2 * np.sin(x0 - 1) - 0.2

        g_1 = d_x * (pow(d_x, 2) + pow(d_y, 2)) ** (-0.5)
        g_2 = d_y * (pow(d_x, 2) + pow(d_y, 2)) ** (-0.5)

        alpha = 0.5

        x0 = x; y0 = y

        x = x0 - g_1*alpha;    y = y0 - g_2*alpha

        if f(x, y) > f(x0, y0):
            x, y, x0, y0, alpha = cond(x, y, x0, y0, -alpha)
            #    Условие остановки
        if np.sqrt(pow(x - x0, 2) + pow(y - y0, 2)) < eps:
            break

    else:

        F_m.append(f(x, y))

        X_m.append(x)
        Y_m.append(y)

        print('x = ', x, '\ty = ', y, '\tf(x, y) = ', f(x, y), '\t| № итерации: ', i, '\nШаг h = ',alpha, '\n')

        print('--------------------------------------------')


    x0 = x;  y0 = y



min_x = round(x0, 2)
min_y = round(y0, 2)

print('Найденная точка минимума: (',  min_x, min_y, ").", ' Число итераций: ', i)


# Диапазон изменения аргументов
x1_plt = np.arange(-2, 2, 0.1)        # x1
x2_plt = np.arange(-0.5, 0.5, 0.1)        # x2

# Формируем функцию 2-х переменных F(x1, x2)
F_plt = np.array([[f(a, b) for a in x1_plt] for b in x2_plt])
fig = plt.figure(figsize=(15, 11), dpi = 110)     # масштаб фигуры
ax = plt.axes(projection='3d')
# создаём прямоугольную сетку из массив x1_plt, x2_plt для построения графика
x1, x2 = np.meshgrid(x1_plt, x2_plt)    # координатные матрицы из координатных векторов
ax.plot_surface(x1, x2, F_plt, color='y', alpha=0.5)


# Подписываем оси
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Ф')
    # Задаём точку
point = ax.scatter(min_x, min_y, f(min_x, min_y), c='red', s=50, label = 'Найденная точка минимума')
plt.plot(X_m, Y_m, F_m, '-*g', label = 'Направление минимизации')
plt.legend()
plt.show()
