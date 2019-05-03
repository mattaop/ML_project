from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model
import random
import numpy as np

from load_data import load_data_detection

random.seed(100)

model_weights = 'weights/character_detection_weights.hdf5'


def data_processing(x):
    """Process data"""
    x = x.astype('float32')
    x /= 255  # Make between 0 and 1
    x = rgb2gray(x)  # Convert to grey scale
    x = x.reshape((1,) + x.shape + (1,))  # Reshape the images
    return x


def check_if_all_zero(img):
    """Check if the image contains under 50% of white or black spots, and if so return True"""
    img = img.reshape(400)
    percentage_zero = 0
    for i in range(len(img)):
        if img[i] < 10**(-5):
            percentage_zero += 1
        elif img[i] > 0.999999:
            percentage_zero += 1
    if percentage_zero/len(img) < 0.5:
        return True
    else:
        return False


def slicing_window(img):
    """Runs over the input image, in slices of 20x20, and predicts character using model_weights"""
    stride_width = 5
    stride_height = 5
    width = 20
    height = 20
    model = load_model(model_weights)
    predictions = []
    # Predict characters
    for i in range(0, img.shape[1]-height, stride_height):
        for j in range(0, img.shape[2]-width, stride_width):
            predicted_prob = model.predict(img[:, i:i + height, j:j + width])
            if np.amax(predicted_prob) > 0.99 and np.argmax(predicted_prob) != 0 and \
                    check_if_all_zero(img[:, i:i + height, j:j + width]):
                predictions.append([np.argmax(predicted_prob), j, i])
                print("Predicted character: ", str(chr(96+np.argmax(predicted_prob))), ", Coordinate: ", j, i)

    # Plot rectangles where characters are detected
    fig, ax = plt.subplots(1)
    ax.imshow(img[0, :, :, 0], cmap='gray')
    for i in range(len(predictions)):
        rect = patches.Rectangle((predictions[i][1], predictions[i][2]), 20, 20,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def character_detection():
    """Load images and detect characters"""
    img1, img2 = load_data_detection()
    img1 = data_processing(img1)
    img2 = data_processing(img2)
    slicing_window(img1)
    slicing_window(img2)


if __name__ == "__main__":
    character_detection()
