import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#import splitfolders
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator, array_to_img
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.keras import Input
from keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB7, EfficientNetV2L, MobileNetV2, ResNet50, \
    ResNet101, InceptionV3, DenseNet121
import tensorflow.python.keras.layers

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time
import datetime

# from tqdm.notebook import tqdm
from tqdm import tqdm
from bokeh.io import output_notebook, show, push_notebook
from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import MISSING_RENDERERS

silence(MISSING_RENDERERS, True)

import sklearn.metrics as metrics
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# print(tf.__version__)
# print(np.__version__)
# print(cv2.__version__)

# Set GPU memory growth to minimize GPU memory consumption
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu_device in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_device, True)


def resize_dataset_and_overwrite(img_size, root, dataset):
    img_cols = img_size[0]
    img_rows = img_size[1]

    img_files = glob.glob(root + '/' + dataset + '/*/*.JPEG')

    # Convert all images for Mobilenet format (224, 224)
    for fn in tqdm(img_files):
        image = cv2.imread(fn)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(fn, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def augment_dataset_with_distortions(root, dataset):
    print("Augmenting dataset with distorted variants [", root + '/' + dataset, "]")

    # Augment training
    img_files = glob.glob(root + '/' + dataset + '/*/*.JPEG')

    # Define ranges of the image distortion to include in the generator
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')

    for fn in tqdm(img_files):
        filename = fn.split('/')[-1]
        img = load_img(fn)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the same directory as the original image
        destdir = fn[:-len(filename)]

        n_aug = 3
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=destdir,
                                  save_prefix='aug_',
                                  save_format='JPEG'):
            i += 1
            if i > n_aug:
                break


#
# Read dataset and labels
#
def read_dataset_and_labels(root, dataset, target_size):
    print("Reading dataset from [" + root + '/' + dataset + "] as ", dataset)
    print(" - Image shape to be used (w,h,c) = ", target_size)

    dataset_files = glob.glob(root + '/' + dataset + '/*/*.JPEG')
    num_classes = len(glob.glob(root + '/' + dataset + '/*'))

    print(" - Number of classes", num_classes)
    print(" - Number of images", len(dataset_files))

    w, h, c = target_size  # width, height, channels

    n_images = len(dataset_files)
    imgs = np.ndarray((n_images, w, h, c))
    labels = np.ndarray((n_images))

    idx = 0
    for fn in tqdm(dataset_files):
        # class
        cat = fn.split('\\')[-2]

        img = load_img(fn, grayscale=False, target_size=(w, h))
        img = img_to_array(img)

        imgs[idx] = img / 255  # rescale to [0,1] range
        labels[idx] = int(cat)

        idx = idx + 1

    # Convert images to single precision and convert categorical labels to
    # vectors: 0 to [1, 0, 0, ...], 1 to [0, 1, 0, ...] etc.

    # Convert images to float
    imgs = imgs.astype('float32')

    # Convert labels to categorical
    labels = tf.keras.utils.to_categorical(labels, num_classes)

    return imgs, labels


#
# Monitoring of the training process with plots updated every epoch showing
# history of the optimization proccess i.e. loss and accuracy for both training and testing
# datasets.
#
class TrainingPlot(tf.keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

        output_notebook()
        self.p1 = figure(width=450, height=300, title='Losses')
        self.p2 = figure(width=450, height=300, title='Accuracy')

        self.target = show(row(self.p1, self.p2), notebook_handle=True)

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 0:
            N = np.arange(0, len(self.losses))

            self.p1.line(N, self.losses, color='blue', legend_label='Training')
            self.p1.line(N, self.val_losses, color='red', legend_label='Testing')

            self.p2.line(N, self.acc, color='blue', legend_label='Training')
            self.p2.line(N, self.val_acc, color='red', legend_label='Testing')

            self.p1.legend.location = "top_left"
            self.p2.legend.location = "top_left"

            push_notebook(handle=self.target)


#
# Evaluate model's accuracy
#
def evaluate_model(model, test_img, y_test):
    score = model.evaluate(test_img, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


#
# Plot random sample of image for spot checking
#
def plot_random_sample(n, x, y):
    n_elements = len(x)

    n_col = 5
    n_row = n // n_col
    if n % n_col > 0:
        n_row += 1

    f = plt.figure(figsize=(12, 6))

    subplot_idx = 1
    for i in range(0, n):
        idx = random.randint(0, n_elements - 1)
        label_str = str(np.argmax(y[idx]))
        f.add_subplot(n_row, n_col, subplot_idx, title=label_str)
        plt.imshow(x[idx])
        subplot_idx += 1

    plt.tight_layout()
    plt.show()


#
# Function to equalize the histogram of colored image
#
def equalize_colored_histogram(img):
    # Change colorspace to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


#
# Change of the gamma value of the image
#
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


#
# Generate uniformly gamma augmented set
#
def dataset_augment_with_gamma_variations(path):
    dirs = glob.glob(path + '/*')

    for dir_name in sorted(dirs):

        print('augmenting gamma -- working in - ', dir_name.split('/')[-1])

        fns = glob.glob(dir_name + '/*')

        # Loop over files
        for fn in tqdm(fns):

            # read image
            img = cv2.imread(fn)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # equalize histogram
            img_eq = img  # equalize_colored_histogram(img)

            # Loop over gamma changes
            for new_gamma in [0.25, 0.5, 0.75, 1.0, 1.125, 1.25, 1.5]:
                #             for new_gamma in [0.125, 0.25, 0.5, 0.75, 1.0, 1.125, 1.25, 1.5, 2.0]:

                img_gamma = adjust_gamma(img_eq, gamma=new_gamma)

                new_fn = dir_name + '/' + fn.split('/')[-1][:-4] + '-gamma-' + str(new_gamma) + '.png'
                cv2.imwrite(new_fn, cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB))


def train():
    # Variants distribution,, load into each sub topic class, but before dont
    # @forget to prepare the image into classification folder, you can refer to code share here:
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    # @ you only need to change the subtopic folder path, forexample ./Ximage12/blur/blurwithbackgournd, it will spilt your image folder into train/eval folder 80%/20%
    #splitfolders.ratio("sub_topic_class_name", output="classification", seed=1337, ratio=(.8, .2,),
    #                   group_prefix=None)  # default values
    # Then don't forget to change the folder name into 0.1.2.3.4...12. because later int(n120102) wont make sense, int(str) or int(float) wont work

    dataset_root = './Ximage12/classification'

    dataset_files = glob.glob(dataset_root + '/' + 'train' + '/0/*.JPEG')
    # num_classes = len(glob.glob(dataset_root + '/' + 'train' + '/*'))
    print(" - Number of images", len(dataset_files))
    # n_images = len(dataset_files)

    batch_size = 128  # if you want you can change here as well
    epochs = 200
    num_classes = 12

    resize_and_overwrite_original_files = False
    augment_with_distortions = False
    augment_with_gamma = False
    target_size = (224, 224, 3)

    # @ you only need to change the model name: ( EfficientNetB0  / EfficientNetB7 / EfficientNetV2L / MobileNetV2 /
    # ResNet50 / ResNet101 / DenseNet121/)
    ##TODO: change the model name
    model_ = EfficientNetB0(input_shape=target_size, weights=None, classes=num_classes)
    model_.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model_.summary()
    solver = tf.keras.optimizers.Adam(learning_rate=0.0001)

    log_dir = os.path.join(
        "logs",
        "fit",
        ##Todo: change the model name
        "EfficientNetB0",
        datetime.datetime.now().strftime("%Y%m%d-%H%M")
    )

    print("Path log:", log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if resize_and_overwrite_original_files:
        print('-- Resizing and overwriting -- ')
        resize_dataset_and_overwrite(target_size, dataset_root, 'train_folder')
        resize_dataset_and_overwrite(target_size, dataset_root, 'eval_folder')
    if augment_with_gamma:
        print('-- Augmenting with gamma changes --')
        dataset_augment_with_gamma_variations(dataset_root + '/train_folder')
        dataset_augment_with_gamma_variations(dataset_root + '/testing')

    if augment_with_distortions:
        print('-- Augmenting with distortions --')
        augment_dataset_with_distortions(dataset_root, 'train_folder')
        augment_dataset_with_distortions(dataset_root, 'testing')

    print('-- Reading datasets --')
    train_img, y_train = read_dataset_and_labels(dataset_root, 'train', target_size)
    test_img, y_test = read_dataset_and_labels(dataset_root, 'val', target_size)

    ##TODO: change the model name
    mcp_save = tf.keras.callbacks.ModelCheckpoint('./EX1_effinetb0_model_200epoch', save_best_only=True,
                                                  monitor='val_loss', mode='min')

    print('-- Training -- ')
    plot_losses = TrainingPlot()
    hist = model_.fit(train_img, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                      validation_data=(test_img, y_test), shuffle=True,
                      callbacks=[plot_losses, tensorboard_callback, mcp_save])

    # @ You only need to change the model name accordingly with what model you trained
    ##TODO: change the model name
    filename = 'EX1_effinetb0_model_200epoch'

    model_.save(filename)

    print('-- Evaluating -- ')


train()
