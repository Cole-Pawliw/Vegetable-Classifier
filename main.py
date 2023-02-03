import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import os

path = "./Vegetable Images"

# reading the images from the folder
name_list = []  # creating a name list for the images

train_list = []  # creating an image list for original images
train_class = []  # creating a list of classes for the training data
validate_list = []  # creating an image list for validate images
validate_class = []  # creating a list of classes for the validate data
test_list = []  # creating an image list for testing images
test_class = []  # creating a list of classes for the testing data

curr_class = 0
for folder_name in os.listdir(path + "/train"):
    current_location = path + "/train/" + folder_name
    name_list.append(folder_name)
    for filename in os.listdir(current_location):
        img = cv2.imread(os.path.join(current_location, filename))  # reading every image in the file
        img = img[:, :, ::-1]
        # apply preprocessing: If things aren't working, try removing this stuff
        img_r = cv2.resize(img, (244, 244))  # resizing to 244x244
        img_nor = cv2.normalize(img_r, None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)  # normalizing using min/max
        train_list.append(img_nor)
        train_class.append([curr_class])
    curr_class += 1
train_list = np.array(train_list)
train_class = np.array(train_class)

curr_class = 0
for folder_name in os.listdir(path + "/validation"):
    current_location = path + "/validation/" + folder_name
    for filename in os.listdir(current_location):
        img = cv2.imread(os.path.join(current_location, filename))  # reading every image in the file
        img = img[:, :, ::-1]
        # apply preprocessing: If things aren't working, try removing this stuff
        img_r = cv2.resize(img, (244, 244))  # resizing to 244x244
        img_nor = cv2.normalize(img_r, None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)  # normalizing using min/max
        validate_list.append(img_nor)
        validate_class.append([curr_class])
    curr_class += 1

validate_list = np.array(validate_list)
validate_class = np.array(validate_class)

curr_class = 0
for folder_name in os.listdir(path + "/test"):
    current_location = path + "/test/" + folder_name
    for filename in os.listdir(current_location):
        img = cv2.imread(os.path.join(current_location, filename))  # reading every image in the file
        img = img[:, :, ::-1]
        # apply preprocessing: If things aren't working, try removing this stuff
        img_r = cv2.resize(img, (244, 244))  # resizing to 244x244
        img_nor = cv2.normalize(img_r, None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)  # normalizing using min/max
        test_list.append(img_nor)
        test_class.append([curr_class])
    curr_class += 1
test_list = np.array(test_list)
test_class = np.array(test_class)

print('Done reading the images')

model = models.Sequential()
model.add(layers.Conv2D(16, (9, 9), activation='relu', input_shape=(244, 244, 3)))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(32, (9, 9), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(32, (9, 9), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(15))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_list, train_class, epochs=10,
                    validation_data=(validate_list, validate_class))

# plot accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# plot loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_list,  test_class, verbose=2)
print(test_acc)
print(test_loss)
