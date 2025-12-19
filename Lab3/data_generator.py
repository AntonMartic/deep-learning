import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#-------------------------------
# Data generator class
#-------------------------------
class DataGenerator:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.x_train = None
        self.dataset = None

    # Generate data
    def generate(self, dataset='mnist'):
        self.x_train = dataset

        # MNIST dataset, provided through Keras
        if dataset == 'mnist':
            (x_train, _), (_, _) = keras.datasets.mnist.load_data()

        # CIFAR10 dataset, provided through Keras
        elif dataset == 'cifar10':
            (x_train, _), (_, _) = keras.datasets.cifar10.load_data()
            
        else:
            raise Exception("Unknown dataset", dataset) 

        # Normalizatie to [-1,1]
        x_train = x_train.astype("float32")
        self.x_train = (x_train - 127.5) / 127.5

        # Add channel dimension if missing
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, -1)
        
        self.x_train = x_train

        # Metadata for plotting/printing
        self.K = 10
        self.C = self.x_train.shape[3]

        if self.verbose:
            print('Data specification:')
            print(f'\tDataset type:           {self.dataset}')
            print(f'\tNumber of channels:     {self.C}')
            print(f'\tTraining data shape:    {self.x_train.shape}')
        
        return self.x_train
        
    # Show some training samples
    def plot(self, xx=12, yy=3, save_path=None):
        plt.figure(figsize=(18, yy * 2))
        # Use gray map if 1 channel, otherwise standard color
        cm = 'gray' if self.C == 1 else None 
        
        for i in range(xx * yy):
            plt.subplot(yy, xx, i + 1)
            # Rescale from [-1, 1] to [0, 1] for imshow
            img = (self.x_train[i] + 1) / 2.0
            
            if self.C == 1:
                plt.imshow(img[:, :, 0], cmap=cm)
            else:
                plt.imshow(img)
            plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
