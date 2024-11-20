import datetime
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2

def plot(img, img2, img3, txt1, txt2, txt3):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(txt1)
    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title(txt2)
    ax[2].imshow(img3, cmap='gray')
    ax[2].set_title(txt3)
    plt.show()

def salvar(imagem, nome):
    print("Desenha salvar a imagem? (s/N)")
    entrada = input()
    if entrada == "s":
        time = datetime.datetime.now().strftime("%Y-%m-%d")
        imagem.save(f'imgs/{nome}{time}.png')

def subtrair(img1, img2):
    npImg1 = np.array(img1)
    npImg2 = np.array(img2)

    return np.abs(npImg1 - npImg2)

def threshold(img, value):
    npImg = np.array(img)
    npImg[npImg < value] = 0
    npImg[npImg >= value] = 255

    return Image.fromarray(npImg)

def diferenca(img1, img2, th=50):
    return threshold(subtrair(img1, img2), th)

def main():
    pcb = Image.open('5-Desafio Stanford/imgs/pcb.png')
    pcbDefected = Image.open('5-Desafio Stanford/imgs/pcbDefected.png')
    
    dif = diferenca(pcb, pcbDefected, 20)
    plot(pcb, pcbDefected, dif, "PCB", "PCB Defeituosa", "Diferença")
    
if __name__ == "__main__":
    main()
