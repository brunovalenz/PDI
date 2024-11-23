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

def threshold(imagem, tipo):
    imagem = imagem.convert('L')
    imagem = np.array(imagem)
    copia = imagem.copy()

    match tipo:
        case 'global':
            limiar = 127
            for i in range(copia.shape[0]):
                for j in range(copia.shape[1]):
                    if copia[i][j] >= limiar:
                        copia[i][j] = 255
                    else:
                        copia[i][j] = 0
        case 'otsu':
            limiar = cv2.threshold(copia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            copia = limiar
        case 'adaptiveMean':
            copia = cv2.adaptiveThreshold(copia, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        case 'adaptiveGaussian':
            copia = cv2.adaptiveThreshold(copia, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        case 'blur':
            copia = cv2.GaussianBlur(copia, (5, 5), 0)
            copia = cv2.threshold(copia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return copia

def main():
    text = Image.open('10-Segmentação2/imgs/hand_text.tif')

    plot(text, threshold(text, 'global'), 'Original', 'Threshold Global')
    plot(text, threshold(text, 'otsu'), 'Original', 'Threshold Otsu')
    plot(text, threshold(text, 'adaptiveMean'), 'Original', 'Threshold Adaptive Mean')
    plot(text, threshold(text, 'adaptiveGaussian'), 'Original', 'Threshold Adaptive Gaussian')
    plot(text, threshold(text, 'blur'), 'Original', 'Threshold Blur')

if __name__ == "__main__":
    main()
