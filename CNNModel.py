import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32
NUM_CLASSES = 100  # Assuming 100 age categories

# Load pretrained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('C:\\Users\\gagan\\Desktop\\Fine tune CNN model', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory('C:\\Users\\gagan\\Desktop\\Fine tune CNN model', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory('C:\\Users\\gagan\\Desktop\\Fine tune CNN model', target_size=IMAGE_SIZE, batch_size=1, class_mode=None, shuffle=False) # Ensure deterministic ordering for visualization

# Train the model
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // BATCH_SIZE, epochs=2, validation_data=validation_generator, validation_steps=validation_generator.samples // BATCH_SIZE)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot sample images with predicted ages
num_images = 1
test_generator.reset()  # Reset generator to start from the beginning
for i in range(num_images):
    image = next(test_generator)  # Get the next image batch
    age_prediction = model.predict(image)
    predicted_age = np.argmax(age_prediction)

    
    # Plot the image
    plt.imshow(image[0])
    plt.title(f"Predicted Age: {predicted_age}")
    plt.axis('off')
    plt.show()

# Save the model
model.save('C:\\Users\\gagan\\Desktop\\Fine tune CNN model\\age_detection_model.h5')

