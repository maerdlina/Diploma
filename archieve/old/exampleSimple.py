import numpy as np
import matplotlib.pyplot as plt

# Свойства материалов (модуль Юнга в Па)
materials = {
    'сталь': 210e9,  # Модуль Юнга для стали
    'алюминий': 70e9,  # Модуль Юнга для алюминия
    'бетон': 30e9,  # Модуль Юнга для бетона
}


def element_stiffness(E, A, L):
    """Вычисляет матрицу жесткости элемента в локальной системе координат."""
    return (E * A / L) * np.array([[1, -1], [-1, 1]])


def assemble_global_stiffness(n_elements, E, A, L):
    """Составляет глобальную матрицу жесткости всей конструкции."""
    n_nodes = n_elements + 1
    K_global = np.zeros((n_nodes, n_nodes))

    for i in range(n_elements):
        K_e = element_stiffness(E, A, L)
        idx = [i, i + 1]
        K_global[np.ix_(idx, idx)] += K_e

    return K_global


def apply_boundary_conditions(K_global, F, fixed_nodes):
    """Учитывает закрепления в глобальной матрице жесткости и векторе нагрузок."""
    for node in fixed_nodes:
        K_global[node, :] = 0
        K_global[:, node] = 0
        K_global[node, node] = 1  # Условие жесткости
        F[node] = 0  # Нагрузка на закрепленных узлах


def plot_beam(displacements_initial, displacements_optimal, n_elements, L):
    """Отрисовывает начальную и оптимизированную балки."""
    x = np.linspace(0, L, n_elements + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Начальная балка
    ax1.plot(x, displacements_initial, marker='o', label='Начальная балка', color='blue')
    ax1.set_title('Начальное состояние балки')
    ax1.set_xlabel('Длина балки (м)')
    ax1.set_ylabel('Перемещение (м)')
    ax1.axhline(0, color='black', lw=0.5, ls='--')
    ax1.grid()
    ax1.legend()

    # Оптимизированная балка
    ax2.plot(x, displacements_optimal, marker='o', label='Оптимизированная балка', color='green')
    ax2.set_title('Оптимизированное состояние балки')
    ax2.set_xlabel('Длина балки (м)')
    ax2.set_ylabel('Перемещение (м)')
    ax2.axhline(0, color='black', lw=0.5, ls='--')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show()


def besos_optimization(n_elements, E, A, L, q, fixed_nodes, iterations=10):
    """Метод топологической оптимизации BESO."""
    A_initial = A
    A_optimal = A_initial * np.ones(n_elements)  # Начальная площадь поперечного сечения

    for _ in range(iterations):
        # Сборка глобальной матрицы жесткости
        K_global = assemble_global_stiffness(n_elements, E, A_optimal.mean(), L / n_elements)

        # Приведение нагрузок к узловым
        F = np.full(n_elements + 1, q * (L / n_elements))  # Узловые нагрузки
        apply_boundary_conditions(K_global, F, fixed_nodes)

        # Решение системы уравнений: u = K^(-1) * P
        displacements = np.linalg.solve(K_global, F)

        # Оптимизация площади поперечного сечения
        for i in range(n_elements):
            if displacements[i] < 0:  # Проверяем перемещение для каждого элемента
                A_optimal[i] *= 0.9  # Уменьшаем площадь

    return A_optimal


def main():
    # Ввод данных от пользователя
    material = input("Введите материал (сталь, алюминий, бетон): ").strip().lower()
    if material not in materials:
        print("Неизвестный материал. Пожалуйста, выберите из списка.")
        return

    # Ввод начальных параметров
    L = float(input("Введите длину балки (м): "))
    q = float(input("Введите равномерно распределенную нагрузку (Н/м): "))

    # Параметры балки
    h = 0.1  # Высота балки (м)
    E = materials[material]  # Модуль Юнга выбранного материала
    n_elements = 4  # Количество элементов
    n_nodes = n_elements + 1  # Количество узлов
    A_initial = h * h  # Начальная площадь поперечного сечения

    # Сборка глобальной матрицы жесткости
    K_global = assemble_global_stiffness(n_elements, E, A_initial, L / n_elements)

    # Приведение нагрузок к узловым
    F = np.full(n_nodes, q * (L / n_elements))  # Узловые нагрузки

    # Учет закреплений (например, узлы 0 и 4)
    fixed_nodes = [0, n_nodes - 1]
    apply_boundary_conditions(K_global, F, fixed_nodes)

    # Решение системы уравнений: u = K^(-1) * P
    displacements_initial = np.linalg.solve(K_global, F)

    # Оптимизация с использованием BESO
    A_optimal = besos_optimization(n_elements, E, A_initial, L, q, fixed_nodes)

    # Сборка глобальной матрицы жесткости после оптимизации
    K_global_optimal = assemble_global_stiffness(n_elements, E, A_optimal.mean(), L / n_elements)
    F_optimal = np.full(n_nodes, q * (L / n_elements))  # Узловые нагрузки
    apply_boundary_conditions(K_global_optimal, F_optimal, fixed_nodes)

    # Решение системы уравнений после оптимизации
    displacements_optimal = np.linalg.solve(K_global_optimal, F_optimal)

    # Визуализация начального и оптимизированного состояния
    plot_beam(displacements_initial, displacements_optimal, n_elements, L)

    # Вывод результатов
    print("\nГлобальная матрица жесткости начального состояния:")
    print(K_global)
    print("\nПеремещения узлов начального состояния:")
    print(displacements_initial)
    print("\nОптимизированная площадь поперечного сечения:")
    print(A_optimal)
    print("\nПеремещения узлов оптимизированного состояния:")
    print(displacements_optimal)


if __name__ == "__main__":
    main()
