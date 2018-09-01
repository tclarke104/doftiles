from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

import numpy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# format the x training and test sets
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# format the y training and test labels
y_test = np_utils.to_categorical(y_test, 10)
y_train = np_utils.to_categorical(y_train, 10)

model = Sequential()

model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),
                activation='relu',
                input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(4,4), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(10, kernel_size=(1,1), activation='softmax'))
model.add(Flatten())

print(model.output_shape)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=32, nb_epoch=10, verbose=1)
