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

def inverter(imagem):
    copia = imagem.copy()
    return 255 - copia

def MPP(imagem, detalhes=100):
    detalhes = 1/detalhes
    copia = imagem.copy()
    new = copia.copy()*0.5
    new = cv2.cvtColor(new.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(copia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = detalhes * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(new, [approx], -1, (255, 0, 0), 3)
    return new

def main():
    amassado = Image.open('Representação/imgs/amassado.png')

    debug = False

    setup = threshold(amassado, 'adaptiveGaussian')
    if debug: 
        plot(amassado, setup, 'Original', 'Threshold Global')
    setup = cv2.morphologyEx(setup, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if debug: 
        plot(amassado, setup, 'Original', 'Threshold Global')
    setup = inverter(setup)
    if debug: 
        plot(amassado, setup, 'Original', 'Threshold Global')
    
    plot(amassado, MPP(setup, 200), "Original", "MPP")

if __name__ == "__main__":
    main()
