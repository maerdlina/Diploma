import cv2
import numpy as np
from stl import mesh

def check_color(pixel):
    gray_value = np.mean(pixel)
    if gray_value > 240:  # Белый цвет
        return 0  # Пустота
    elif 220 < gray_value <= 240:  # Серый
        return 1  # Материал
    elif gray_value <= 220:  # Темный серый до черного
        return 1  # Материал
    return 0  # Неизвестно

def analyze_image(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    data = np.zeros((height, width), dtype=int)

    for y in range(height):
        for x in range(width):
            data[y, x] = check_color(image[y, x])

    return data

def create_stl(data, filename, beam_width=1, beam_height=1):
    height, width = data.shape
    vertices = []
    faces = []

    for y in range(height):
        for x in range(width):
            if data[y, x] == 1:  # Если это материал
                # Добавляем вершины для объемной балки
                base_index = len(vertices)

                # Нижняя плоскость
                vertices.append([x, y, 0])  # Нижний левый
                vertices.append([x + beam_width, y, 0])  # Нижний правый
                vertices.append([x + beam_width, y + beam_height, 0])  # Верхний правый
                vertices.append([x, y + beam_height, 0])  # Верхний левый

                # Верхняя плоскость
                vertices.append([x, y, 1])  # Нижний левый (верх)
                vertices.append([x + beam_width, y, 1])  # Нижний правый (верх)
                vertices.append([x + beam_width, y + beam_height, 1])  # Верхний правый (верх)
                vertices.append([x, y + beam_height, 1])  # Верхний левый (верх)

                # Создаем грани
                faces.append([base_index, base_index + 1, base_index + 2])
                faces.append([base_index, base_index + 2, base_index + 3])
                faces.append([base_index + 4, base_index + 5, base_index + 6])
                faces.append([base_index + 4, base_index + 6, base_index + 7])
                faces.append([base_index, base_index + 1, base_index + 5])
                faces.append([base_index, base_index + 5, base_index + 4])
                faces.append([base_index + 1, base_index + 2, base_index + 6])
                faces.append([base_index + 1, base_index + 6, base_index + 5])
                faces.append([base_index + 2, base_index + 3, base_index + 7])
                faces.append([base_index + 2, base_index + 7, base_index + 6])
                faces.append([base_index + 3, base_index, base_index + 4])
                faces.append([base_index + 3, base_index + 4, base_index + 7])

    vertices = np.array(vertices)
    faces = np.array(faces)

    model = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            model.vectors[i][j] = vertices[f[j]]

    model.save(filename)

# Пример использования
image_data = analyze_image('optimization_result.png')
create_stl(image_data, 'output_model.stl', beam_width=1, beam_height=1)
