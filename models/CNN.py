import numpy as np
import time
import matplotlib.pyplot as plt
import argparse

from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D, add, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

from feature_extraction.data_processing import scale_input, add_dimension, grey_scale, \
    invert_colors, add_pictures_without_chars
from load_data import load_data_chars
import os

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNN:
    """Class containing multiple convolutional neural networks"""
    def __init__(self, num_classes, sample, network_type='residual', model_weights=None):
        self.num_classes = num_classes
        self.input_shape = sample.shape
        self.model_weights = model_weights
        if model_weights:
            print('Loading model from ', model_weights, '...')
            self.model = load_model(model_weights)
        else:
            print('Initializing new ', network_type, ' model...')
            if network_type == 'residual':
                self.model = self.residual_network()
            elif network_type == 'simple':
                self.model = self.simple_network()
            else:
                NameError(network_type, ' is not a model.')

    def simple_network(self):
        """Simple network structure"""
        input_img = Input(shape=self.input_shape)

        #####################
        # Convolution layer #
        #####################

        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_img)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        ##########################
        # Fully connected layers #
        ##########################

        x = Dense(512, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def residual_network(self):
        """CNN structure, with residual learning"""
        input_img = Input(shape=self.input_shape)

        ######################
        # Convolution layers #
        ######################

        def residual_layers(input_layer):
            y = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       activation='relu')(input_layer)
            y = BatchNormalization()(y)
            y = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       activation='relu')(y)
            y = add([y, input_layer])
            y = BatchNormalization()(y)
            return y

        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_img)
        x = BatchNormalization()(x)
        x = residual_layers(x)
        x = residual_layers(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = residual_layers(x)
        x = residual_layers(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)

        ##########################
        # Fully connected layers #
        ##########################

        x = Dense(256, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, x, y, model_weights):
        """Train model using data generator to augment the data"""
        data_generator = ImageDataGenerator(rotation_range=20,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            width_shift_range=2,
                                            height_shift_range=2,
                                            preprocessing_function=invert_colors)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
        mcp_save = ModelCheckpoint(model_weights, save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [early_stopping, mcp_save]
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
        data_generator.fit(x_train)
        training_history = self.model.fit_generator(data_generator.flow(x_train, y_train, batch_size=64),
                                                    steps_per_epoch=len(x_train) / 64, epochs=200, callbacks=callbacks,
                                                    validation_data=(x_val, y_val), verbose=2)
        self.plot_training_history(training_history)
        model = load_model(model_weights)
        test_history = model.evaluate(x=x, y=y)
        print("Train loss: ", test_history[0], ", train accuracy: ", test_history[1])

    def test(self, x, y, model_weights):
        """Loads the best weights, and tests the model"""
        model = load_model(model_weights)
        history = model.evaluate(x=x, y=y)
        print("Test loss: ", history[0], ", test accuracy: ", history[1])
        return history[1]

    def predict_character(self, x):
        prediction = self.model.predict(x=x)
        return prediction

    def plot_training_history(self, history):
        """Plots training history"""
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def plot_predictions(self, x, y, model_weights):
        """Plots some random predictions, both good and bad"""
        model = load_model(model_weights)
        predictions = model.predict(x=x)
        predicted_chars = np.argmax(predictions, axis=1)
        right_predictions = []
        right_predicted = []
        wrong_predictions = []
        wrong_predicted = []
        for i in range(len(predicted_chars)):
            if predicted_chars[i] == np.argmax(y[i]) and len(right_predicted) < 4:
                right_predictions.append(i)
                right_predicted.append(predicted_chars[i])
            if predicted_chars[i] != np.argmax(y[i]) and len(wrong_predicted) < 4:
                wrong_predictions.append(i)
                wrong_predicted.append(predicted_chars[i])

        for i in range(len(right_predictions)):
            im = x[right_predictions[i]].reshape([20, 20])
            plt.imshow(im, cmap='gray')
            plt.title("Predicted character: " + str(chr(97 + right_predicted[i])) + ", real character: " +
                      str(chr(97 + np.argmax(y[right_predictions[i]]))))
            plt.legend()
            plt.show()

        for i in range(len(wrong_predictions)):
            im = x[wrong_predictions[i]].reshape([20, 20])
            plt.imshow(im, cmap='gray')
            plt.title("Predicted character: " + str(chr(97 + wrong_predicted[i])) + ", real character: " +
                      str(chr(97 + np.argmax(y[wrong_predictions[i]]))))
            plt.legend()
            plt.show()


def fit_cnn(x, y, model_weights='weights/model_weights.hdf5', network_type='simple', trials=1):
    print('=== Convolution Neural Network ===')
    test_accuracy = np.zeros(trials)
    running_time = np.zeros(trials)
    x = scale_input(x)
    x = grey_scale(x)
    x = add_dimension(x)
    # x, y = add_pictures_without_chars(x, y)
    y = to_categorical(y, int(np.max(y)+1))
    for i in range(trials):
        print('Training network ', i + 1)
        start = time.time()
        random_state = 100 + i
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=random_state,
                                                            stratify=y)
        network = CNN(num_classes=len(y[0]), sample=x_train[0], network_type=network_type)
        network.train(x_train, y_train, model_weights)
        test_accuracy[i] = network.test(x_test, y_test, model_weights)
        # network.plot_predictions(x_test, y_test, model_weights)
        running_time[i] = time.time() - start
        print('Running time: ', running_time[i])
    print('Average test accuracy over ', trials, ' trials: ', np.mean(test_accuracy))
    print('Average running time over ', trials, ' trials: ', np.mean(running_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convolution neural network')
    parser.add_argument('-n', action="store", dest="network_type", default='simple')
    arg_input = parser.parse_args()
    img, target = load_data_chars()
    fit_cnn(img, target, trials=1, network_type=arg_input.network_type)
