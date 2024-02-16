import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

# Define where the shapes are
working_dir = 'shapes'
classes = ['circle', 'square', 'star', 'triangle']

# Create a data generator with 20% for validation
data_generator = ImageDataGenerator(validation_split=0.2)

# Define the training generator
training_generator = data_generator.flow_from_directory(
    working_dir,
    classes=classes,
    batch_size=100,
    subset='training',
    color_mode='grayscale'
)
# Define the validation generator
validation_generator = data_generator.flow_from_directory(
    working_dir,
    classes=classes,
    batch_size=100,
    subset='validation',
    color_mode='grayscale'
)

# # Check to see what the images look like interpreted
images, labels = training_generator.next()
# fig, axes = plt.subplots(1, 4, figsize=(20, 20))
# axes = axes.flatten()
# axes_idx = 0
# for one_hot_label in range(4):
#     for image, label in zip(images, labels):
#         if label[one_hot_label] == 1:
#             ax = axes[axes_idx]
#             ax.imshow(image[:, :, 0], cmap='Greys_r')
#             ax.axis('off')
#             ax.set_title(classes[one_hot_label])
#             axes_idx += 1
#             break
# plt.tight_layout()
# plt.show()
#
# # Testing different representations
first_image_in_batch = images[0]
image_shape = first_image_in_batch.shape
# print(image_shape)
# plt.imshow(first_image_in_batch[:, :, 0], cmap='Greys_r')
# print(first_image_in_batch[:, :, 0])

# Create a convolutional layer and add 16 filters of 5 by 5 pixels
convolutional_layer = Conv2D(16, (5, 5), activation='relu', input_shape=image_shape)

# Create a pooling filter to reduce the image sizes
max_pool_layer = MaxPool2D()

# Create a flatten layer to flatten the results of the pool layer into a single vector
flatten_layer = Flatten()

# Create a dense layer to act as the output for the flattened data
dense_layer = Dense(4, activation='softmax')

# Build the model
cnn_model = Sequential([
    convolutional_layer,
    max_pool_layer,
    flatten_layer,
    dense_layer
])

cnn_model.summary()

# Train the model
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = cnn_model.fit(training_generator, epochs=5)

# Test the model after training
val_loss, val_acc = cnn_model.evaluate(validation_generator)

print(f"Evaluation result on Test Data : Loss = {val_loss}, accuracy = {val_acc}")

# See what images were falsely predicted
predictions = cnn_model.predict(validation_generator)

predicted_labels = np.argmax(predictions, axis=1)

ground_truth_labels = validation_generator.classes

misclassified_indices = np.where(predicted_labels != ground_truth_labels)[0]

# Display misclassified images in pages
images_per_page = 9
num_misclassified = len(misclassified_indices)
num_pages = (num_misclassified + images_per_page - 1) // images_per_page

for page in range(num_pages):
    plt.figure(figsize=(15, 15))
    for i in range(images_per_page):
        index = page * images_per_page + i
        if index >= num_misclassified:
            break
        plt.subplot(3, 3, i + 1)
        img = plt.imread(validation_generator.filepaths[misclassified_indices[index]])
        plt.imshow(img, cmap='gray')
        plt.title(f'Predicted: {classes[predicted_labels[misclassified_indices[index]]]}, '
                  f'Actual: {classes[ground_truth_labels[misclassified_indices[index]]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
