from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('always')

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='fine')
train_images, test_images = train_images / 255.0, test_images / 255.0
model_100 = models.Sequential()
model_100.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_100.add(layers.MaxPooling2D((2, 2)))
model_100.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_100.add(layers.MaxPooling2D((2, 2)))
model_100.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_100.add(layers.Flatten())
model_100.add(layers.Dense(64, activation='relu'))
model_100.add(layers.Dense(100)) 
model_100.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_100 = model_100.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))


(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='coarse')
train_images, test_images = train_images / 255.0, test_images / 255.0
model_20 = models.Sequential()
model_20.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_20.add(layers.MaxPooling2D((2, 2)))
model_20.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_20.add(layers.MaxPooling2D((2, 2)))
model_20.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_20.add(layers.Flatten())
model_20.add(layers.Dense(64, activation='relu'))
model_20.add(layers.Dense(20))
model_20.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history_20 = model_20.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))

acc_100 = history_100.history['accuracy']
val_acc_100 = history_100.history['val_accuracy']
acc_20 = history_20.history['accuracy']
val_acc_20 = history_20.history['val_accuracy']
epochs_range = range(1, len(acc_100) + 1)
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc_100, label='Обучение на 100 классах')
plt.plot(epochs_range, acc_20, label='Обучение на 20 классах')
plt.legend(loc='lower right')
plt.title('Точность обучения')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_acc_100, label='Валидация на 100 классах')
plt.plot(epochs_range, val_acc_20, label='Валидация на 20 классах')
plt.legend(loc='lower right')
plt.title('Точность валидации')
plt.show()

y_pred_100 = model_100.predict(test_images)
y_pred_labels_100 = np.argmax(y_pred_100, axis=1)
report_100 = classification_report(test_labels, y_pred_labels_100)
print("Отчет о классификации для 100 классов:\n", report_100)

y_pred_20 = model_20.predict(test_images)
y_pred_labels_20 = np.argmax(y_pred_20, axis=1)
report_20 = classification_report(test_labels, y_pred_labels_20)
print("Отчет о классификации для 20 классов:\n", report_20)

model_100.save('100.keras')
model_20.save('20.keras')