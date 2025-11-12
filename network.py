import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.datasets import fetch_openml
mnist = fetch_openml(name="mnist_784")

print(mnist.keys())

dict_keys = {'data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'}

data = mnist.data
labels = mnist.target

n = np.random.choice(np.arrange(data.shape[0]+1))

print(n)