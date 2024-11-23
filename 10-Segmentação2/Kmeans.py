import datetime
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
import math

def plot(img, edit, txt1, txt2):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(txt1)
    ax[1].imshow(edit, cmap='gray')
    ax[1].set_title(txt2)
    plt.show()

def salvar(imagem, nome):
    print("Desenha salvar a imagem? (s/N)")
    entrada = input()
    if entrada == "s":
        time = datetime.datetime.now().strftime("%Y-%m-%d")
        imagem.save(f'imgs/{nome}{time}.png')

def kmeans(imagem, k=8):
    imagem = np.array(imagem)
    copia = imagem.copy()
    
    Z = np.float32(imagem.reshape((-1,3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((imagem.shape))

    cores = np.array((
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [0, 0, 0],
        [255, 255, 255]
    )
    , dtype=float)

    res = cores[label.flatten()]
    res3 = res.reshape((imagem.shape))

    return res2, res3


def main():
    sushi = Image.open('10-Segmentação2/imgs/sushi.jpg')

    a, b = kmeans(sushi, 8)
    plot(sushi, a, "Original", "Kmeans")
    plot(sushi, b, "Original", "Kmeans")

if __name__ == "__main__":
    main()
