import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import TextBox, Button
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import zoom
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use('TkAgg')  # Установка бэкенда

# Глобальная переменная для остановки оптимизации
stop_optimization = False

def beso(nu, length, width, volfrac, penal, rmin, fx, fy, load_value):
    global stop_optimization
    stop_optimization = False

    Emin = 1e-9
    Emax = 1.0

    nelx = int(length * 50)  # Увеличиваем плотность сетки
    nely = int(width * 50)  # Увеличиваем плотность сетки
    ndof = 2 * (nelx + 1) * (nely + 1)

    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0

    KE = lk_programmatically(nu)
    edofMat, iK, jK = generate_edof_and_stiffness(nelx, nely)
    H, Hs = filter_matrix(nelx, nely, rmin)

    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    fixed = np.union1d(dofs[0:2 * (nely + 1):2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    free = np.setdiff1d(dofs, fixed)

    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    load_dof = 2 * ((nely + 1) * fx + fy) + 1
    f[load_dof, 0] = load_value

    # Создаем окно для графика и кнопки
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        zoom(xPhys.reshape((nely, nelx)), 2, order=1),
        cmap='gray',
        interpolation='bicubic',
        norm=colors.Normalize(vmin=-1, vmax=0)
    )
    ax.xaxis.tick_top()
    ax.set_title("Оптимизация топологии")
    fig.colorbar(im, ax=ax, orientation="horizontal")

    # Добавляем кнопку "Остановить оптимизацию" под графиком
    stop_button_ax = fig.add_axes([0.4, 0.01, 0.2, 0.05])
    stop_button = Button(stop_button_ax, "Остановить оптимизацию")

    # Функция обработки нажатия кнопки
    def stop_optimization_callback(event):
        global stop_optimization
        stop_optimization = True
        print("Оптимизация остановлена пользователем.")
        show_completion_message("Оптимизация завершена!")

    stop_button.on_clicked(stop_optimization_callback)

    plt.ion()
    fig.show()

    loop = 0
    change = 1
    min_change_threshold = 1e-4
    min_change_count = 0
    max_min_change_count = 10
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    while loop < 100:
        if stop_optimization:
            print("Оптимизация остановлена пользователем.")
            break

        loop += 1

        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]
        u[free, 0] = spsolve(K, f[free, 0])

        # Sensitivity analysis
        ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)).sum(1)

        # Update sensitivities for both removal and addition
        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)).reshape(nely * nelx) * ce
        dv[:] = np.ones(nely * nelx)

        # Apply filtering
        dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)

        xold[:] = x
        (x[:], g) = optimal(nelx, nely, x, volfrac, dc, dv, g)

        # Update physical design variable
        xPhys[:] = x

        change = np.linalg.norm(x.reshape(nely * nelx, 1) - xold.reshape(nely * nelx, 1), np.inf)

        if change < min_change_threshold:
            min_change_count += 1
        else:
            min_change_count = 0

        if min_change_count >= max_min_change_count:
            print("Минимальные изменения обнаружены, завершение...")
            break

        im.set_array(zoom(-xPhys.reshape((nelx, nely)).T, 2, order=1))
        plt.pause(0.01)
        fig.canvas.draw()

        if loop % 5 == 0:
            plt.savefig(f'iteration_{loop}.png', dpi=300)

    plt.ioff()
    xx = x.reshape(nely, nelx, order='F')

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.grid(False)
    im.set_array(zoom(-xx, 2, order=1))
    fig.canvas.draw()
    plt.show()

    # Отображаем сообщение по завершению
    show_completion_message("Оптимизация завершена!")

# Функция для вывода сообщения о завершении
def show_completion_message(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Завершение", message)

def create_input_interface():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # Поля для ввода
    ax.text(0.00, 0.9, "Введите параметры балки", fontsize=14, transform=ax.transAxes)

    # Изображение после текста
    img = plt.imread("../beam_image.png")  # Убедитесь, что изображение "beam_image.png" находится в рабочем каталоге
    ax_img = plt.axes([0.2, 0.6, 0.5, 0.2])  # Создаем ось для изображения
    ax_img.imshow(img)
    ax_img.axis('off')

    # Поле для длины
    ax.text(0.05, 0.55, "Длина балки (м):", fontsize=12, transform=ax.transAxes)
    length_ax = plt.axes([0.15, 0.48, 0.3, 0.04])
    length_input = TextBox(length_ax, '', initial='2')

    # Поле для ширины
    ax.text(0.5, 0.55, "Ширина балки(м):", fontsize=12, transform=ax.transAxes)
    width_ax = plt.axes([0.5, 0.48, 0.3, 0.04])
    width_input = TextBox(width_ax, '', initial='1')

    # Поле для нагрузки
    ax.text(0.05, 0.35, "Величина нагрузки (Н):", fontsize=12, transform=ax.transAxes)
    load_ax = plt.axes([0.45, 0.37, 0.3, 0.04])
    load_input = TextBox(load_ax, '', initial='10')

    # Функция отображения балки
    def show_beam():
        try:
            length = float(length_input.text)
            width = float(width_input.text)
            load_value = float(load_input.text)

            if length <= 0 or width <= 0 or load_value <= 0:
                raise ValueError("Длина, ширина балки и нагрузка должны быть положительными.")

            fig_beam, ax_beam = plt.subplots(figsize=(8, 4))
            ax_beam.set_xlim(-0.5, length + 0.5)
            ax_beam.set_ylim(-width - 0.5, 0.5)
            ax_beam.plot([0, length], [0, 0], color='blue', lw=4)
            ax_beam.plot([0, length], [-width, -width], color='blue', lw=4)
            ax_beam.plot([0, 0], [0, -width], color='blue', lw=4)
            ax_beam.plot([length, length], [0, -width], color='blue', lw=4)
            ax_beam.annotate(
                '', xy=(0, 0), xytext=(0, 0.5),
                arrowprops=dict(facecolor='red', shrink=0.05)
            )
            ax_beam.text(0.100, 0.15, f'Нагрузка: {load_value}Н', fontsize=12, color='red')
            ax_beam.set_aspect('equal', adjustable='box')
            ax_beam.axis('off')
            ax_beam.set_title("Отображение балки")
            plt.show()
        except ValueError as e:
            tk.Tk().withdraw()
            messagebox.showerror("Ошибка ввода", str(e))

    def start_optimization():
        try:
            length = float(length_input.text)
            width = float(width_input.text)
            load_value = float(load_input.text)

            if length <= 0 or width <= 0 or load_value <= 0:
                raise ValueError("Длина, ширина балки и нагрузка должны быть положительными.")

            plt.close(fig)
            beso(nu=0.3, length=length, width=width, volfrac=0.4, penal=3.0, rmin=1.5, fx=0, fy=0, load_value=load_value)
        except ValueError as e:
            tk.Tk().withdraw()
            messagebox.showerror("Ошибка ввода", str(e))

    # Кнопка "Отобразить балку"
    button_ax1 = plt.axes([0.1, 0.1, 0.35, 0.1])
    button_show_beam = Button(button_ax1, "Отобразить балку")
    button_show_beam.on_clicked(lambda event: show_beam())

    button_ax2 = plt.axes([0.55, 0.1, 0.35, 0.1])
    button_optimize = Button(button_ax2, "Запуск оптимизации")
    button_optimize.on_clicked(lambda event: start_optimization())

    plt.show()

def lk_programmatically(nu):
    E = 1
    a = 1 / 2 - nu / 6
    b = 1 / 8 + nu / 8
    c = -1 / 4 - nu / 12
    d = -1 / 8 + 3 * nu / 8
    e = -1 / 4 + nu / 12
    f = -1 / 8 - nu / 8
    g = nu / 6
    h = 1 / 8 - 3 * nu / 8
    KE = E / (1 - nu ** 2) * np.array([
        [a, b, c, d, e, f, g, h],
        [b, a, h, g, f, e, d, c],
        [c, h, a, f, g, d, e, b],
        [d, g, f, a, h, c, b, e],
        [e, f, g, h, a, b, c, d],
        [f, e, d, c, b, a, h, g],
        [g, d, e, b, c, h, a, f],
        [h, c, b, e, d, g, f, a]
    ])
    return KE

def generate_edof_and_stiffness(nelx, nely):
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3])
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    return edofMat, iK, jK

def filter_matrix(nelx, nely, rmin):
    nfilter = int(nelx * nely * (2 * (np.ceil(rmin) - 1) + 1) ** 2)
    iH = np.zeros(nfilter, dtype=int)
    jH = np.zeros(nfilter, dtype=int)
    sH = np.zeros(nfilter)
    k = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k1 in range(max(i - int(np.ceil(rmin)) + 1, 0), min(i + int(np.ceil(rmin)), nelx)):
                for k2 in range(max(j - int(np.ceil(rmin)) + 1, 0), min(j + int(np.ceil(rmin)), nely)):
                    col = k1 * nely + k2
                    fac = rmin - np.sqrt((i - k1) ** 2 + (j - k2) ** 2)
                    if fac > 0:
                        iH[k] = row
                        jH[k] = col
                        sH[k] = fac
                        k += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)
    return H, Hs

def optimal(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(0.001,
                          np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        if np.sum(xnew) - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew, g

if __name__ == "__main__":
    create_input_interface()
