import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels) = mnist.load_data()
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
training_images = training_images / 255.0 # the range is between 0 to 255 so we normalized the values 
test_images = test_images/255.0           # because computer works a lot better with 0 and 1.
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128,activation=tf.nn.relu),  # activation function tells neurons what to do
                                    # relu return 0 or max number(x>0) and pass it to next layer.
                                    tf.keras.layers.Dense(10,activation=tf.nn.softmax)]) # softmax returns the maximum value as 1 and all other values as 0.
                                    # so softmax save time for finding maximum value. we only need to find 1 and that's it.
model.compile(optimizer = tf.keras.optimizers .Adam(),  # Adam = replacement of sgd 
              loss = 'sparse_categorical_crossentropy', # SCC and CC
              metrics = ['accuracy'])
model.fit(training_images,training_labels,epochs = 5)
model.evaluate(test_images,test_labels)
