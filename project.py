import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import tensorflow as tf


tf.random.set_seed(25)

ALPHABET = [1, 2, 5, 10, 20]
TRAIN_DIR = "data/train/"
TEST_DIR = "data/test/"
ROTATION_ANGLE = 6
# ROTATION_ANGLE = 18  # 100% 15 epochs
# ROTATION_ANGLE = 24  # 99.9%
IMG_WIDTH = 192
IMG_HEIGHT = 192
# 178


def load_train_res(file_name):
    d = pd.read_csv(file_name, delimiter=",", engine="python")
    x = d.iloc[:, 0]
    y = d.iloc[:, 1]
    return [[xi, yi] for xi, yi in zip(x, y)]


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 10, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    plt.imshow(image, 'gray')
    plt.show()


def dilate(image, iterations):
    kernel = np.ones((5, 5))
    return cv2.dilate(image, kernel, iterations=iterations)


def erode(image, iterations):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=iterations)


def resize_region(region):
    return cv2.resize(region, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)


def select_roi_single(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cropped_image = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 10000 and (y != x != 0):
            cropped_image = resize_region(image_orig[y:y + h, x:x + w])
    return cropped_image


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


def convert_output(data):
    nn_outputs = []
    for index in range(len(data)):
        output = np.zeros(len(ALPHABET))
        for i in range(len(ALPHABET)):
            if ALPHABET[i] == data[index]:
                output[i] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_model(input_shape, output_size):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, activation="relu"))
    # model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape, activation="relu"))
    model.add(Conv2D(24, kernel_size=(3, 3), activation='relu'))
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    # model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    # model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))  # just a try
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation='softmax'))
    return model


def train_network(model, x_train, y_train, epochs):
    x_train = np.asarray(x_train, np.float32).reshape(len(x_train), IMG_WIDTH, IMG_HEIGHT, 1)
    y_train = np.asarray(y_train, np.float32)

    print("\nTraining started...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=1, shuffle=True)
    print("\nTraining completed...")
    return model


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        print(winner(output))
        print(len(alphabet))
        result.append(alphabet[winner(output)])
    return result


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def load_training_data(train_res):
    return [load_image(TRAIN_DIR + image_name) for (image_name, value) in train_res]


def crop_training_data(raw):
    data = []
    for image in raw:
        bin_input_image = image_bin(image_gray(image))
        cropped_image = select_roi_single(image.copy(), dilate(bin_input_image, 3))
        data.append(image_gray(cropped_image))
    return data


def expand_data_with_rotated_images(cropped_data):
    data = []
    for input_image in cropped_data:
        data.append(input_image)
        for i in range(ROTATION_ANGLE, 360, ROTATION_ANGLE):
            data.append(rotate_image(input_image, i))
    return data


def get_expected_outputs(res):
    data = []
    for (img_name, value) in res:
        data.extend([value] * int(360 / ROTATION_ANGLE))
    return data


def get_training_data(res):
    print("Preparing data...")
    inputs_raw = load_training_data(res)
    inputs_cropped = crop_training_data(inputs_raw)
    inputs = expand_data_with_rotated_images(inputs_cropped)
    outputs = get_expected_outputs(res)
    x_train = [scale_to_range(image) for image in inputs]
    y_train = convert_output(outputs)
    return x_train, y_train


def train(res):
    x_train, y_train = get_training_data(res)
    model = create_model(output_size=len(ALPHABET), input_shape=(IMG_WIDTH, IMG_HEIGHT, 1))
    return train_network(model, x_train, y_train, epochs=10)


def main():
    train_res = load_train_res(TRAIN_DIR + "res.csv")
    model = train(train_res)

    model.save("./model")
    # load_model("./model")

    # for i in range(1, 12):
    #     input_image = load_image("dataProject2/" + str(i) + ".jpg")
    #     bin_input_image = image_bin(image_gray(input_image))
    #     selected_regions, numbers = select_roi(input_image.copy(), (dilate(invert(bin_input_image))))
    #     display_image(bin_input_image)
    #     display_image(selected_regions)
    #     inputs = prepare_for_ann(numbers)
    #     result = model.predict(np.array(inputs, np.float32))
    #     print(result)
    #     print("\n")
    #     print(display_result(result, alphabet))


if __name__ == "__main__":
    main()
