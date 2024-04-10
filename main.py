import zipfile
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Concatenate
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Unzip the dataset
zip_path = 'basicfinal.zip'
extract_dir = 'basicfinal/'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Step 2: Image Preprocessing and Augmentation
def add_noise(img):
    variance = 0.01 * np.random.uniform() * 255
    noise = np.random.normal(0, variance, img.shape)
    img_noisy = img + noise
    return img_noisy

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=add_noise,
    brightness_range=[0.2,1.0])

test_val_datagen = ImageDataGenerator()

# Step 3: Define the CNN Model
def create_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)
    concatenated = Concatenate()([x, x])
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(concatenated)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Step 4: Prepare Data Loaders
batch_size = 128
train_generator = train_datagen.flow_from_directory(
    extract_dir + 'Train/',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_val_datagen.flow_from_directory(
    extract_dir + 'Val/',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_val_datagen.flow_from_directory(
    extract_dir + 'Test/',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Step 5: Train the Model
model = create_cnn_model((128, 128, 3), num_classes=50)  # 11 vowels + 39 consonants = 50 classes
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=40,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# Step 6: Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# Step 7: Plot Training Loss and Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

# Step 8: Generate and Plot Confusion Matrix
predictions = model.predict(test_generator, steps=test_generator.samples // batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes[:len(predicted_classes)]
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
