import mnist_loader
import numpy as np
import pdb
from matplotlib import pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network2_new as network2
 
#~ net = network2.Network([3, 2, 1], 
#~ cost=network2.CrossEntropyCost)

#~ x = np.array([[1],[2],[3]])
#~ y = np.array([[1]])

#~ result = net.backprop(x,y)



net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost, activations=['sigmoid','sigmoid'])
net.large_weight_initializer()
net.SGD(training_data, 5, 10, 0.5,
    evaluation_data=test_data,
    lmbda = 0.1, # this is a regularization parameter
    monitor_evaluation_cost=False,
    monitor_evaluation_accuracy=False,
    monitor_training_cost=False,
    monitor_training_accuracy=False,
    regularizer="l1")




#~ pixels = first_image.reshape((28, 28))
#~ plt.imshow(pixels, cmap='gray')
#~ plt.show()