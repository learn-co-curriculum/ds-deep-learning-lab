
# Zalando example

What does this all cover?
**General (dense layers)**
- Keras
- Multilayer perceptron
- activation functions
- loss function
- matrix notation practice
- hyperparameter testing (trial and error)
- regularization via early stopping (plot validation vs training accuracy)

**CNN**
- convolutional neural nets
- use of all layers: convolution layers, pooling layers, Fully connected layers
- sense of the general layer structure
- show pictures in python
- Visualizing intermediate convnet outputs 
- regularization via dropout
- make a class prediction

**Things I have not done (yet)**
- k-fold cross-validation (yet wonder about the added value at this stage, makes comp time longer. Or should make the data smaller)
- having jpeg images and actually making them manageable to work with (is this done earlier?)
- data augmentation --> need de jpegs to do so
- vanishing/exploding gradients
- L2 regularization
---> sequence processing with convolutional neural networks?


# 1. Introduction

In this example we'll get a sense of just how powerful convolutional neural networks get! This data set, the so-called `Fashion-MNIST` was constructed by a research team at online fashion retailer Zalando.  Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.


```python
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image  as img
```


```python
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
```


```python
train_images, train_labels = load_mnist( path = "data_zalando/",kind='train')
test_images, test_labels = load_mnist(path = "data_zalando/", kind='t10k')
```

This is a fairly big data set. If we keep it as such, runtimes will be longer. Let's take samples of this data set, so we work with 10,000 training instances and 2,500 test instances. Do remember that working with bigger data is generally always better when it comes to training models and getting more accurate results, longer computation times are pretty much the only downside!


```python
from random import *
import random
index_train = range(0,59999)
index_test= range(0,9999)

random.seed(1109)
train_sample = sample(index_train,  10000)
test_sample = sample(index_test,  2500)
```


```python
train_images = train_images[train_sample]
train_labels = train_labels[train_sample]
test_images = test_images[test_sample]
test_labels = test_labels[test_sample]
```

Let's verify the shape of the train_images and test_images


```python
np.shape(train_images)
```


```python
np.shape(test_images)
```

You can see that now, there are 10,000 observations in the training set, each representing 1 image of 28 x 28 pixels, so each observation contains 784 inputs that range between [0,255] (for a greyscale image, going from white to black). Similarly, there are 2500 images in the test set. Let's have a look at a random observation from `train_images`, let's say, the 4th observation.


```python
train_images[3]
```

It would be nice if we could actually see what this picture looks like. Note that the form $(n\_obs, total\_n\_pixels)$, which is what we obtained looking at `np.shape(train_images)`, is desirable when training neural networks, but if we want to *visualize* the actual images, we need to reshape $total\_n\_pixels$ to something of the form ($horiz\_pixels$, $vertical\_pixels$). Let's use the numpy function `matrix.reshape` to look at the 4th image.


```python
reshape_example= np.matrix.reshape(train_images[3],28,28)
```

Let's quickly have a look at how this changed the data structure. Notice how you have an 28 x 28 matrix now!


```python
reshape_example
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(reshape_example, cmap=plt.cm.binary)
plt.show()
```

Remember that the resolution of this image is pretty low, but for this example it's definitely clear that this is a shirt!

Let's look at the labels. As mentioned, there are 10 classes of clothing in this data set. In the `train_labels` (`test_labels`) data set, this means that there are exactly 10,000 (2500) inputs with each input ranging from [0,9].


```python
np.shape(train_labels)
```


```python
train_labels
```

Although we'll keep this decoded version to train and test our model (actually, we'll decode even further and perform one-hot encoding), it would be nice if we could do a sanity check on our data now and verify that the train instances are labeled correctly. The dictionary `cat_dict` contains the so-called "encoding information", to encode the data back from number to clothing category. The `encoded_labels` object contains the encoded labels. Let's have a look at the 4th observation in the `encoded_labels` and check if the picture we showed before is indeed identified as a dress.


```python
encoded_labels = {0:'T-shirt/Top',
                  1:'Trouser',
                  2:'Pullover',
                  3:'Dress',
                  4:'Coat',
                  5:'Sandal',
                  6:'Shirt',
                  7:'Sneaker',
                  8:'Bag',
                  9:'Ankle Boot'}

encoded_train = [encoded_labels[i] for i in train_labels]
encoded_test = [encoded_labels[i] for i in test_labels]
```


```python
encoded_train[3]
```

This label seems to be correct! By replacing the value passed in `index_test` below, you can look at several pictures and the associated label. 


```python
index_fill = 1376 # replace with any value between 0 and 9999

reshape= np.matrix.reshape(train_images[index_fill],28,28)

import matplotlib.pyplot as plt
plt.imshow(reshape, cmap=plt.cm.binary)
plt.show()

encoded_train[index_fill]
```

# 2. Build densely connected network as a baseline

Every time you run a newer, more advanced model on your data, you would want to make sure you compare it with what you have learned. For that reason, let's look at a dense neural network as a baseline before diving into convolutional networks. As data is manipulated slightly differently in convolutional networks, let's perform data manipulation for the dense network, and we'll start over from scratch for the convolutional neural network.

Remember that the image data is stored in an array of shape (10000, 784) of type uint8 with values in the [0, 255] interval. To serve as input for the model, we want transform it into a float32 array of the same shape, but with values between 0 and 1 instead of 0 and 255.  


```python
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
```

We're not quite done yet. Remember that instead of a label somewhere [0,9], neural networks generally use one-hot encoding. For this exercise, this means that the label for each observation is replaced by a vector containing 9 `0`'s and just 1 `1`, in the place of the label index.

You can use `to_categorical` in `keras.utils` to do this.


```python
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels= to_categorical(test_labels)
```

From our 10,000 training set elements, let set 2000 aside for validation.


```python
img_val = train_images[:2000]
partial_img_train  = train_images[2000:]
label_val = train_labels[:2000]
partial_label_train = train_labels[2000:]
```

Now, let's build our baseline network. You'll build two dense hidden layers, the first one with 128 hidden units, and the second one with 64 hidden units. Use the rectified linear unit as an activation function for both. Remember that the input shape is 28 x 28 for both layers. The last layer of the network should have a softmax activation function (because this is a multiclass problem), and 10 units.

Let's train the model for 50 epochs in batches of 128.


```python
from keras import models
from keras import layers
dense = models.Sequential()
dense.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
dense.add(layers.Dense(64, activation='relu', input_shape=(28 * 28,)))
dense.add(layers.Dense(10, activation='softmax'))

dense.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

dense_fit = dense.fit(partial_img_train, 
                      partial_label_train, 
                      epochs=50, 
                      batch_size=128,
                      validation_data=(img_val, label_val))
```

Let's plot the evolution of the training versus validation loss along the epochs.


```python
hist_dense = dense_fit.history
loss_values = hist_dense['loss']
val_loss_values = hist_dense['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'g.', label='Training loss')
plt.plot(epochs, val_loss_values, 'g', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Let's plot the evolution of the training versus validation accuracy along the epochs.


```python
acc_values = hist_dense['acc'] 
val_acc_values = hist_dense['val_acc']

plt.plot(epochs, acc_values, 'r.', label='Training acc')
plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

The validation accuracy and loss seem to be fairly volatile, but as we move along the epochs, the validation loss seems to go up again.
It seems appropriate to stop training after about ~20 epochs. Let's run the model again changing `epochs = 20`, and let's look at the results.


```python
dense_final = models.Sequential()
dense_final.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
dense_final.add(layers.Dense(64, activation='relu', input_shape=(28 * 28,)))
dense_final.add(layers.Dense(10, activation='softmax'))

dense_final.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

dense_final_fit = dense_final.fit(partial_img_train, 
                                  partial_label_train, 
                                  epochs=20, 
                                  batch_size=128,
                                  validation_data=(img_val, label_val))
```


```python
results_dense_final = dense_final.evaluate(test_images, test_labels)
```


```python
results_dense_final
```

The test set classification accuracy is about 80%. Not bad, for a 10-fold classification and a fairly small data size. Let's see if we can do better with a convolutional net!

# 3. A convolutional network

## 3.1 Data pre-processing

The data needs to be pre-processed differently. Let's start again from scratch.


```python
train_images, train_labels = load_mnist( path = "data_zalando/",kind='train')
test_images, test_labels = load_mnist(path = "data_zalando/", kind='t10k')

import random
index_train = range(0,59999)
index_test= range(0,9999)

random.seed(1109)
train_sample = sample(index_train,  10000)
test_sample = sample(index_test,  2500)

train_images = train_images[train_sample]
train_labels = train_labels[train_sample]
test_images = test_images[test_sample]
test_labels = test_labels[test_sample]
```

Where the input layer for a dense neural network is fine to be $(n\_obs, total\_n\_pixels)$, we want the training in data in convolutions to be $(n\_obs, horiz\_pixels, vertical\_pixels, num\_channels)$, because convolutions operate over 3D tensors, with two spatial axes (height and width) and a depth axis (also referred to as $channels$ axis). For an RGB image, the dimension of the depth axis is 3, because of the three color channels red, green, and blue. For a black-and-white picture the depth is 1 (levels of gray). We need to reshape train_images and test_images as follows:


```python
train_images = train_images.reshape((10000, 28, 28, 1))
test_images = test_images.reshape((2500, 28, 28, 1))
```

Remember that the image data is stored in an array of shape (10000, 784) of type uint8 with values in the [0, 255] interval. To serve as input for the convolutional model, you'll need to transform it again into a float32 array of the same shape, but with values between 0 and 1 instead of 0 and 255.  


```python
test_images = test_images.astype('float32') / 255
train_images = train_images.astype('float32') / 255
```

Let's perform one-hot encoding on the labels again! You can use `to_categorical` in `keras.utils` to do this.


```python
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```


```python
train_labels
```

Again, we'll want to split the training set up in actual training data and validation data.


```python
img_val = train_images[:2000]
partial_img_train  = train_images[2000:]
label_val = train_labels[:2000]
partial_label_train = train_labels[2000:]
```

Let's look at exactly what convolutions do. A convolution starts by sliding "patches" over the input. Patches are typically 3 x 3 or 5 x 5. Because of patching, the output width and height will differ from the input and depend on 2 things:
- border effects
- The use of strides


The input shape for the first layer is (28, 28, 1), as previously discussed. For a convolutional layer, 2 additional decisions need to be made:
- The depth of the convolutional layer, also the number of filters created by the convolution.
- The dimensions of the so-called "patches" (generally 3 x 3 or 5 x 5).

Using keras, this can be programmed as follows:
`Conv2D(output_depth, (window_height, window_width))`

## 3.2 Running the first model

Use 3 convolutional layers (the first one with output depth 32, the latter two with output depth 64). Make sure all layers use 3x3 patches.

Then, to downsample feature maps use maxpooling layers in between the convolution layers. Use the common 2 × 2 window and stride 2, in order to downsample the feature maps by a factor of 2. 


```python
# A small convnet
from keras import layers 
from keras import models

conv = models.Sequential()
conv.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
conv.add(layers.MaxPooling2D((2, 2)))
conv.add(layers.Conv2D(64, (3, 3), activation='relu')) 
conv.add(layers.MaxPooling2D((2, 2)))
conv.add(layers.Conv2D(64, (3, 3), activation='relu'))

# you can use `padding = valid` and `padding = same`. Strides \= 1 for convolutional layers are not very common
```

When convolutional networks, generally, dense layers are added towards the end of the network. First, we need to flatten the 3D outputs to 1D, and then we can add a few dense layers. Let's add one hidden layer with 64 units and relu, and again a softmax layer towards the end of the exercise.


```python
conv.add(layers.Flatten())
conv.add(layers.Dense(64, activation='relu'))
conv.add(layers.Dense(10, activation='softmax'))
```

Note how the output layers change dimension. A summary of what exaclty is happening can be found using `.summary()`


```python
conv.summary()
```

Let's run this model, using 30 epochs and a batch size of 64. Use both the train data and the validation data in the model.


```python
conv.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
conv_fit = conv.fit(partial_img_train, 
                      partial_label_train, 
                      epochs=30, 
                      batch_size=64,
                      validation_data=(img_val, label_val))
```

Let's plot the evolution of the training versus validation loss along the epochs again.


```python
plt.clf()
hist_conv = conv_fit.history
loss_values = hist_conv['loss']
val_loss_values = hist_conv['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'g.', label='Training loss')
plt.plot(epochs, val_loss_values, 'g', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Let's plot the evolution of the training versus validation loss along the epochs again.


```python
plt.clf()

acc_values = hist_conv['acc'] 
val_acc_values = hist_conv['val_acc']

plt.plot(epochs, acc_values, 'r.', label='Training acc')
plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

The validation accuracy and loss seem to be fairly volatile again, but as we move along the epochs, the validation loss definitely goes up. Here, seems appropriate to stop after about ~14 epochs. Let's retrain the entire model again.


```python
conv_final = models.Sequential()
conv_final.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
conv_final.add(layers.MaxPooling2D((2, 2)))
conv_final.add(layers.Conv2D(64, (3, 3), activation='relu')) 
conv_final.add(layers.MaxPooling2D((2, 2)))
conv_final.add(layers.Conv2D(64, (3, 3), activation='relu'))

conv_final.add(layers.Flatten())
conv_final.add(layers.Dense(64, activation='relu'))
conv_final.add(layers.Dense(10, activation='softmax'))

conv_final.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
conv_final_fit = conv_final.fit(partial_img_train, 
                      partial_label_train, 
                      epochs=14, 
                      batch_size=64,
                      validation_data=(img_val, label_val))
```


```python
results_conv_final = conv_final.evaluate(test_images, test_labels)
```


```python
results_conv_final
```

The test set classification accuracy is 86.3%. Coming from 83.6 in the sequential model this is about a 3% improvement. For bigger data sets this could even be more! Let's see if we can still improve our model!

## 3.3 Using drop-out regularization

Note how we "early stopping" to fight overfitted models. Recall that there is another method, dropout regularization, to counter overfitting. Rerun our `conv` model, yet with a dropout layer right before the densely connected classifier, with parameter 0.5.


```python
conv_drop = models.Sequential()
conv_drop.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
conv_drop.add(layers.MaxPooling2D((2, 2)))
conv_drop.add(layers.Conv2D(64, (3, 3), activation='relu')) 
conv_drop.add(layers.MaxPooling2D((2, 2)))
conv_drop.add(layers.Conv2D(64, (3, 3), activation='relu'))

conv_drop.add(layers.Flatten())
conv_drop.add(layers.Dropout(0.5))
conv_drop.add(layers.Dense(64, activation='relu'))
conv_drop.add(layers.Dense(10, activation='softmax'))

conv_drop.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
conv_drop_fit = conv_drop.fit(partial_img_train, 
                      partial_label_train, 
                      epochs=30, 
                      batch_size=64,
                      validation_data=(img_val, label_val))
```


```python
plt.clf()
hist_conv_drop = conv_drop_fit.history
loss_values = hist_conv_drop['loss']
val_loss_values = hist_conv_drop['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'g.', label='Training loss')
plt.plot(epochs, val_loss_values, 'g', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


```python
plt.clf()

acc_values = hist_conv_drop['acc'] 
val_acc_values = hist_conv_drop['val_acc']

plt.plot(epochs, acc_values, 'r.', label='Training acc')
plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Note how the dropout regularization made the curves much smoother! Additionally, the respective accuracies and losses don't diverge to the extent they did before. Let's stop after 20 epochs.


```python
conv_drop_final = models.Sequential()
conv_drop_final.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
conv_drop_final.add(layers.MaxPooling2D((2, 2)))
conv_drop_final.add(layers.Conv2D(64, (3, 3), activation='relu')) 
conv_drop_final.add(layers.MaxPooling2D((2, 2)))
conv_drop_final.add(layers.Conv2D(64, (3, 3), activation='relu'))

conv_drop_final.add(layers.Flatten())
conv_drop_final.add(layers.Dropout(0.5))
conv_drop_final.add(layers.Dense(64, activation='relu'))
conv_drop_final.add(layers.Dense(10, activation='softmax'))

conv_drop_final.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
conv_drop_final_fit = conv_drop_final.fit(partial_img_train, 
                      partial_label_train, 
                      epochs=20, 
                      batch_size=64,
                      validation_data=(img_val, label_val))
```


```python
results = conv_drop_final.evaluate(test_images, test_labels)
```


```python
results
```

Look at this! You were able to increase the test set accuracy by almost 1% (being 86.3% for the model without dropout regularization).

## 3.4 Making predictions


```python
predictions = conv_drop_final.predict(test_images)
index_test = 2
```


```python
predictions[index_test]
```


```python
test_labels[index_test]
```


```python
reshape= np.matrix.reshape(test_images[index_test],28,28)

%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(reshape, cmap=plt.cm.binary)
plt.show()

encoded_test[index_test]
```

This coat is incorrectly classified as a dress! Not really surprising, looking at it, right?

## 3.5 Try tweaking the model yourself!

The possibilities are literally endless!
- add/remove layers
- In some layers, change `padding = valid` and `padding = same`
- change activation functions
- change optimizer
- change batch size
- change patch dimensions (from 3 x 3 to 5 x 5)
- If you have more time, try running everything on the bigger training set! Do results improve?

## 3.6 Visualizations (WIP, maybe)


```python
index_fill = 1219 # replace by any value between 0 and 9999

reshape= np.matrix.reshape(train_images[index_fill],28,28)

%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(reshape, cmap=plt.cm.binary)
plt.show()

encoded_train[index_fill]
```


```python
from keras import models
layer_outputs = [layer.output for layer in conv_final.layers[:8]] 
activation_model = models.Model(inputs=conv_final.input, outputs=layer_outputs)
```


```python
interm_tensor = np.expand_dims(train_images[index_test], axis=0) 
interm_tensor /= 255
```


```python
interm_tensor
```


```python
activations = activation_model.predict(interm_tensor)
```


```python
first_layer_activation = activations[0]
print(first_layer_activation.shape)
```


```python
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 1], cmap='viridis');
```


```python
plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis');
```
