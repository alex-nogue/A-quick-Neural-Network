# Relevant libraries
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from keras.optimizers import Adam

# Split into train and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_x = x_train.reshape(60000, 784) / 255
val_x = x_test.reshape(10000, 784) / 255

# Visualize an image
from matplotlib import pyplot as plt
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt
gen_image(train_x[1]).show()

# Model 
autoencoder = Sequential()
autoencoder.add(Dense(512,  activation='elu', input_shape=(784,)))
autoencoder.add(Dense(128,  activation='elu'))
autoencoder.add(Dense(10,   activation='linear', name="bottleneck"))
autoencoder.add(Dense(128,  activation='elu'))
autoencoder.add(Dense(512,  activation='elu'))
autoencoder.add(Dense(784,  activation='sigmoid'))
autoencoder.compile(loss='mean_squared_error', optimizer = Adam())
trained_model = autoencoder.fit(train_x, train_x, batch_size=1024, epochs=10, verbose=1, validation_data=(val_x, val_x))
encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
encoded_data = encoder.predict(train_x)  # bottleneck representation
decoded_output = autoencoder.predict(train_x)        # reconstruction
encoding_dim = 10

# Return the decoder
encoded_input = Input(shape=(encoding_dim,))
decoder = autoencoder.layers[-3](encoded_input)
decoder = autoencoder.layers[-2](decoder)
decoder = autoencoder.layers[-1](decoder)
decoder = Model(encoded_input, decoder)

# Add a picture external to MNIST to assess results
%matplotlib inline
from keras.preprocessing import image
img = image.load_img("./Files/photo.jpg", target_size=(28, 28), color_mode = "grayscale")
input_img = image.img_to_array(img)

# Reconstruction error comparison: MNIST vs external picture
def get_reconstruction_error(pic):
    inputs = pic.reshape(1,784)
    target_data = autoencoder.predict(inputs)
    dist = np.linalg.norm(inputs - target_data, axis=-1)
    return dist
get_reconstruction_error(val_x[1])  # MNIST
get_reconstruction_error(input_img) # External picture

# Visually compare the reconstruction of the images
gen_image(val_x[1]).show()      # MNIST
gen_image(autoencoder.predict(val_x[1].reshape(1,784))).show()
gen_image(input_img).show()     # External picture
gen_image(autoencoder.predict(input_img.reshape(1,784))).show()