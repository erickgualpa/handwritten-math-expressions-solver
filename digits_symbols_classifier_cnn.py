import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from DigitsSymbolsDataset import DigitsSymbolsDataset

K.clear_session()

def get_digits_symbols_classifier():
    ## LOAD DATA ######################################################
    ds = DigitsSymbolsDataset()
    (x_train, y_train), (x_test, y_test) = ds.load_data()
    ###################################################################

    ## PREPROCESS DATA ################################################
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Convert  class vectors to binary class matrices
    num_classes = 18
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    ###################################################################

    # MODEL BUILDING ##################################################
    batch_size = 128
    epochs = 2

    # Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    ###################################################################

    # MODEL TRAINING ##################################################
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    print('The model has successfully trained')
    ###################################################################

    # MODEL EVALUATION ################################################
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    ###################################################################

    return model