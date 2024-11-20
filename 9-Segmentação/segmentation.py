import datetime
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2

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

def convolucao(imagem, kernel):
    return scipy.signal.convolve2d(imagem, kernel, mode='same')

def threshold(imagem, limiar):
    copia = imagem.copy()
    # 1 𝑠𝑒 | 𝑅 𝑥, 𝑦 ≥ 𝑇
    # 0 𝑐𝑎𝑠𝑜 𝑜 𝑐𝑜𝑛𝑡𝑟á𝑟𝑖o
    for i in range(copia.shape[0]):
        for j in range(copia.shape[1]):
            if copia[i][j] >= limiar:
                copia[i][j] = 1
            else:
                copia[i][j] = 0
    return copia

def limiarizacao(imagem, limiar):
    copia = imagem.copy()
    for i in range(copia.shape[0]):
        for j in range(copia.shape[1]):
            if copia[i][j] >= limiar:
                copia[i][j] = 255
            else:
                copia[i][j] = 0
    return copia

def canny(imagem, limiar, limiar2):
    return cv2.Canny(imagem, limiar, limiar2)

def um(img):
    # dectecção de pontos
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    img = img.convert('L')
    img = np.array(img)

    edit = convolucao(img, kernel)
    edit = threshold(edit, 100)

    plot(img, edit, "Original", "Pontos")

def dois(img):
    # limiarização
    img = img.convert('L')
    img = np.array(img)
    edit = limiarizacao(img, 100)
    plot(img, edit, "Original", "Limiarização")

def tres(img, limiar, limiar2):
    # Detector de bordas Canny
    img = img.convert('L')
    img = np.array(img)
    edit = canny(img, limiar, limiar2)
    plot(img, edit, "Original", "Canny")

    borrada = cv2.GaussianBlur(img, (5, 5), 0)
    edit = canny(borrada, limiar, limiar2)
    plot(borrada, edit, "Borrada", "Canny")

def main():
    pontos = Image.open('Segmentação/imgs/pontos.jpeg')
    impressao = Image.open('Segmentação/imgs/impressao.tif')
    emanuel = Image.open('Segmentação/imgs/emanuel.jpg')

    um(pontos)

    dois(impressao)

    tres(emanuel, 60, 70)

if __name__ == "__main__":
    main()
