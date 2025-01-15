import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Загрузка и подготовка данных
def load_data():
    # Здесь вы можете загрузить ваши данные (например, изображения и метки)
    # Используйте tf.keras.preprocessing.image_dataset_from_directory для загрузки изображений
    # Для примера создадим случайные данные
    num_samples = 1000
    img_height, img_width = 128, 128
    num_classes = 10  # Количество классов для классификации
    x_train = np.random.rand(num_samples, img_height, img_width, 3)  # Случайные изображения
    y_train = np.random.randint(0, num_classes, num_samples)  # Случайные метки
    return x_train, y_train

# Создание модели CNN
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 10 классов

    return model

# Компиляция и обучение модели
def train_model(model, x_train, y_train):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32)

# Основная функция
if __name__ == "__main__":
    x_train, y_train = load_data()
    model = create_model()
    train_model(model, x_train, y_train)

    # Сохранение модели
    model.save('cnn_model.h5')
