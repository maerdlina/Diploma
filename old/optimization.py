from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use("TkAgg")
import ctypes
import joblib
from PIL import Image
import os

# Загрузка модели для преобразования изображения в STL
def load_stl_converter(filename):
    return joblib.load(filename)

# Функция для преобразования изображения в STL
def convert_image_to_stl(converter, image_path, stl_path):
    try:
        converter.image_to_stl(image_path, stl_path)
        print(f"STL model saved to {stl_path}")
    except Exception as e:
        print(f"Error during STL conversion: {e}")

# MAIN DRIVER
def main(nu, length, width, volfrac, penal, rmin, ft):
    Emin = 1e-9
    Emax = 1.0

    # Определяем количество элементов на основе длины и ширины
    nelx = int(length * 10)  # Например, 10 элементов на единицу длины
    nely = int(width * 10)    # Например, 10 элементов на единицу ширины

    # dofs:
    ndof = 2 * (nelx + 1) * (nely + 1)
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0
    dc = np.zeros((nely, nelx), dtype=float)

    # FE
    KE = lk(nu)
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Фильтрация
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt(((i - k) ** 2 + (j - l) ** 2))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc += 1

    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # Закрепления
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    fixed = np.union1d(dofs[0:2 * (nely + 1):2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    free = np.setdiff1d(dofs, fixed)

    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Нагрузки
    f[1, 0] = -1

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    ax.xaxis.tick_top()
    fig.show()

    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    while change > 0.01 and loop < 2000:
        loop += 1

        # Решение FE
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]
        u[free, 0] = spsolve(K, f[free, 0])

        # Чувствительность
        ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)).sum(1)
        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)).reshape(nely * nelx) * ce
        dv[:] = np.ones(nely * nelx)

        # Фильтрация
        dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)

        # Критерий оптимальности
        xold[:] = x
        (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)
        xPhys[:] = x

        change = np.linalg.norm(x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1), np.inf)

        # Вывод на экран
        im.set_array(-xPhys.reshape((nelx, nely)).T)
        plt.pause(0.01)
        fig.canvas.draw()

    plt.ioff()
    xx = x.reshape(nely, nelx, order='F')
    ctypes.windll.user32.MessageBoxW(0, "Оптимизация завершена!", "Поздравляем", 1)

    # Настройка отображения
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_axisbelow(True)

    # Убираем метки и сетку
    ax.xaxis.set_visible(False)  # Скрыть метки по оси X
    ax.yaxis.set_visible(False)  # Скрыть метки по оси Y
    ax.grid(False)  # Убрать сетку

    # Вывод результата
    im.set_array(-xx)
    fig.canvas.draw()
    fig.set_size_inches(10, 10)
    plt.get_current_fig_manager().window.state('zoomed')

    # Сохранение изображения
    plt.savefig("optimization_result.png", bbox_inches='tight')

    plt.show()
    input()

# Элементы матрицы жесткости
def lk(nu):
    E = 1
    k = np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                         [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                         [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                         [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                         [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                         [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                         [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                         [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE

# Критерий оптимальности
def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    xnew = np.zeros(nelx * nely)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        gt = g + np.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)

if __name__ == "__main__":
    nu = 0.3
    length = float(input("Введите длину балки (в метрах): "))  # Ввод длины
    width = float(input("Введите ширину балки (в метрах): "))   # Ввод ширины
    volfrac = 0.4
    rmin = 5.4
    penal = 3.0
    ft = 0

    # Загрузка модели
    converter = load_stl_converter('image_to_stl_converter.joblib')

    # Путь к изображению и выходному STL файлу
    image_path = 'image_to_stl_converter.joblib'
    stl_path = 'model.stl'

    # Проверка, существует ли изображение
    if os.path.exists(image_path):
        # Выполнение преобразования
        convert_image_to_stl(converter, image_path, stl_path)
    else:
        print(f"Image not found: {image_path}")

    # Запуск основной функции
    main(nu, length, width, volfrac, penal, rmin, ft)
