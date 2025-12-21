import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

def plot_gan_training(log):
    """
    Plots the Generator and Discriminator losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(log.history['d_loss'], label='Discriminator Loss')
    plt.plot(log.history['g_loss'], label='Generator Loss')
    plt.title('GAN Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_generated_digits(generator, latent_dim, n=10):
    """
    Generates and plots a row of digits to see progress.
    """
    random_latent_vectors = tf.random.normal(shape=(n, latent_dim))
    generated_images = generator(random_latent_vectors)
    
    # Rescale from [-1, 1] to [0, 1] for display
    generated_images = (generated_images + 1) / 2.0
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

def plot_manual_history(history_dict):
    plt.figure(figsize=(10, 5))
    plt.plot(history_dict['d_loss'], label="Discriminator Loss")
    plt.plot(history_dict['g_loss'], label="Generator Loss")
    plt.title("GAN Training Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Extraction of weights from Keras model
def get_weights(model):
    W = []
    b = []
    lname = []

    # Types of layers we want to extract
    layer_names = ['conv','pool','flatten','dense']
    
    # Extract weights and biases
    for l in range(len(model.layers)):
        for j in range(len(layer_names)):
            if model.layers[l].name.find(layer_names[j]) >= 0:
                lname.append(layer_names[j])
        Wl = model.layers[l].get_weights()

        # Convolutional kernels and biases for conv layers
        if lname[l] == 'conv':
            W.append(Wl[0])
            b.append(Wl[1])

        # Weight matrix and biases for dense layers
        elif lname[l] == 'dense':
            W.append(np.transpose(Wl[0]))
            b.append(Wl[1][:,np.newaxis])

        # Other layers doesn't contain any weights
        else:
            W.append([])
            b.append([])

    return (W,b,lname)
