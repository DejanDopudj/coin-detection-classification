import os
import sys

import numpy as np
import cv2  # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import collections
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation


from tensorflow.keras.optimizers import SGD

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 75, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    plt.imshow(image, 'gray')
    plt.show()

def dilate(image):
    kernel = np.ones((5, 5))
    return cv2.dilate(image, kernel, iterations=20)

def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 1500 < area and (y != x != 0):
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x - 40, y - 40), ((x + w) + 40,(y + h) + 40), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions

def scale_to_range(image):
    return image / 255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='binary_crossentropy', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        print(winner(output))
        print(len(alphabet))
        result.append(alphabet[winner(output)])
    return result


if __name__ == "__main__":
    values = [3, 6, 6, 9, 8, 8]
    alphabet = [1,5,20,1,5,2,10,10,20,2,20,20,10,10,5,5,2,2,1,1]
    input_image = load_image("dataProject2/" + str(1) + ".jpg")
    bin_input_image = image_bin(image_gray(input_image))
    selected_regions, numbers = select_roi(input_image.copy(), (dilate(invert(bin_input_image))))
    input_image = load_image("dataProject2/" + str(3) + ".jpg")
    bin_input_image = image_bin(image_gray(input_image))
    selected_regions, numbers2 = select_roi(input_image.copy(), (dilate(invert(bin_input_image))))
    # for i in range(1, 7):
    #     print("dataProject2/"+str(i)+".jpg")
    #     input_image = load_image("dataProject2/"+str(i)+".jpg")
    #     bin_input_image = image_bin(image_gray(input_image))
    #     selected_regions, numbers = select_roi(input_image.copy(), (dilate(invert(bin_input_image))))
    inputs = prepare_for_ann(numbers+numbers2)
    outputs = convert_output(alphabet)
    ann = create_ann(output_size=len(alphabet))
    ann = train_ann(ann, inputs, outputs, epochs=5000)
    for i in range(1,12):
        input_image = load_image("dataProject2/"+str(i)+".jpg")
        bin_input_image = image_bin(image_gray(input_image))
        selected_regions, numbers = select_roi(input_image.copy(),(dilate(invert(bin_input_image))))
        display_image(bin_input_image)
        display_image(selected_regions)
        inputs = prepare_for_ann(numbers)
        result = ann.predict(np.array(inputs, np.float32))
        print(result)
        print("\n")
        print(display_result(result, alphabet))