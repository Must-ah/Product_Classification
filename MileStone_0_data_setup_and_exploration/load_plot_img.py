import matplotlib.pylab as plt


def load_img(img_path: str, img_name: str):
    """Loads an image from the given path.
    Args:
        img_path (str): path to the image
        img_name (str): name of the image
    Returns:
        image (np.array): image as a numpy array
    """
    return plt.imread(img_path + '/' + img_name)

import numpy as np
def plot_img(img: np.array):
    """Plots an image.
    Args:
        img (np.array): image as a numpy array
    """
    plt.imshow(img)
    plt.axis('off')
    plt.show()
