import datetime
import numpy as np
from PIL import Image
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

def convolucao(imagem, matriz):
    img = np.array(imagem)

    dif = (matriz.shape[0]-1) // 2

    img = np.pad(img, dif, mode='constant')

    res = img.copy()

    for x in range(dif, img.shape[0]-dif):
        for y in range(dif, img.shape[1]-dif):
            res[x, y] = np.sum(img[x-dif:x+dif+1, y-dif:y+dif+1] * matriz)

    return Image.fromarray(res[dif:img.shape[0]-dif, dif:img.shape[1]-dif])

def matrizMedia(size):
    if((size % 2) == 0):
        size += 1
    if(size < 3):
        size = 3
    return np.full((size, size), 1/(size*size))

def matrizGaussiana(size, sigma): 
    if((size % 2) == 0):
        size += 1
    if(size < 3):
        size = 3
    matriz = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            matriz[x, y] = np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2*sigma**2))
    return matriz / np.sum(matriz)

def main():
    biel = Image.open('imgs/biel.png')
    cameraman = Image.open('imgs/cameraman.tif')
    lena = Image.open('imgs/lena_gray_512.tif')
    imagem = None

    print("Escolha a imagem: ")
    print("1 - biel")
    print("2 - cameraman")
    print("3 - lena")
    entrada = input()
    
    match entrada:
        case "1":
            imagem = biel

        case "2":
            imagem = cameraman

        case "3":
            imagem = lena

        case _:
            print("Opção inválida")
            exit()

    media = matrizMedia(1)

    media = convolucao(imagem, media)

    plot(imagem, media, "Original", "Média")

if __name__ == "__main__":
    main()
