import wget
from pyasn1_modules.rfc8358 import id_ct_xml
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch

import numpy as np
import random
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

def tsne_plot(data):
    tsne = TSNE(n_components=3, random_state=42, perplexity=data.shape[0]-1)
    data_3d = tsne.fit_transform(data)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    num_points = len(data_3d)
    colors = plt.cm.tab20(np.linspace(0, 1, num_points))

    for idx, point in enumerate(data_3d):
        ax.scatter(point[0], point[1], point[2], label=str(idx), c=colors[idx])


    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    ax.set_zlabel('TSNE Component 3')

    plt.title("3D t-SNE Visualization")
    plt.legend(title = "Input Order")
    plt.show()


filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

wget.download(url, filename)

print("file downloaded")

def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    # Split the text into paragraphs (simple split by newline characters)
    paragraphs = text.split('\n')
    # Filter out any empty paragraphs or undesired entries
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs


# Read the text file and split it into paragraphs
paragraphs = read_and_split_text('companyPolicies.txt')
paragraphs[0:10]


