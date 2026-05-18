import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('hand_data.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].astype(str).values

encoder = LabelEncoder()
y = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

print(f"Detected classes: {encoder.classes_}")
print(f"Total dataset shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(63,)),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Training")
model.fit(X_train, y_train, epochs = 60, batch_size = 64, validation_data = (X_test, y_test), callbacks = [early_stop])

print("Exporting model to TFLite...")

model.export('exported_model')

converter = tf.lite.TFLiteConverter.from_saved_model('exported_model')
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
