import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf

image_size = (224, 224)
batch_size = 32
data_path = r'D:\task\task5\archive\food-101\food-101\images'


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(101, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_data, validation_data=val_data, epochs=5)
model.save("food_model.h5")
calorie_dict = {
    'apple_pie': 296,
    'baby_back_ribs': 320,
    'baklava': 334,
    'beef_carpaccio': 240,
    'beef_tartare': 250,
    'beet_salad': 150,
    'beignets': 289,
    'bibimbap': 560,
    'bread_pudding': 310,
    'breakfast_burrito': 450,
    'bruschetta': 190,
    'caesar_salad': 330,
    'cannoli': 280,
    'caprese_salad': 250,
    'carrot_cake': 320
}

def predict_food(image_path):
    model = load_model("food_model.h5")
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array_expanded)
    class_index = np.argmax(result)

    class_labels = list(train_data.class_indices.keys())
    predicted_food = class_labels[class_index]
    calories = calorie_dict.get(predicted_food, "Unknown")

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{predicted_food.replace('_', ' ').title()} - {calories} kcal", fontsize=14)
    plt.show()

    print("Predicted Food:", predicted_food)
    print("Estimated Calories:", calories)


predict_food(r"D:\task\task5\archive\food-101\food-101\images\beef_tartare\173525.jpg")

