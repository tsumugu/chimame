from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np

root_dir = "./image/"
categories = ["chino","maya","megu","other"]
nb_classes = len(categories)
image_size = 50

def main():
    X_train, X_test, Y_train, Y_test = np.load("chimame.npy")
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)

def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def model_train(X, Y):
    model = build_model(X.shape[1:])
    model.fit(X, Y, batch_size=32, nb_epoch=30)
    hdf5_file = "chimame-model.hdf5"
    model.save_weights(hdf5_file)
    return model

def model_eval(model, X, Y):
    score = model.evaluate(X, Y)
    print("loss=",score[0])
    print("accuracy=",score[1])

if __name__ == "__main__":
    main()