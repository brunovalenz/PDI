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

def convolucao(imagem, matriz):
    img = np.array(imagem)

    dif = (matriz.shape[0]-1) // 2

    img = np.pad(img, dif, mode='constant')

    res = img.copy()

    for x in range(dif, img.shape[0]-dif):
        for y in range(dif, img.shape[1]-dif):
            res[x, y] = np.sum(img[x-dif:x+dif+1, y-dif:y+dif+1] * matriz)

    return Image.fromarray(res[dif:img.shape[0]-dif, dif:img.shape[1]-dif])

def convolucaoSciPy(imagem, matriz):
    img = np.array(imagem)
    return scipy.signal.convolve2d(img, matriz, mode='same')

def convolucaoOpenCV(imagem, matriz):
    img = np.array(imagem)
    return cv2.filter2D(img, -1, matriz)

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


def matrizLaplaciana(size):
    if((size % 2) == 0):
        size += 1
    if(size < 3):
        size = 3
    matriz = np.zeros((size, size))
    matriz[size//2, size//2] = -4
    matriz[size//2-1, size//2] = 1
    matriz[size//2+1, size//2] = 1
    matriz[size//2, size//2-1] = 1
    matriz[size//2, size//2+1] = 1
    return matriz

def matrizSobelX():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def matrizSobelY():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def gradiente(imagem):
    sobelX = matrizSobelX()
    sobelY = matrizSobelY()

    gradienteX = convolucaoOpenCV(imagem, sobelX)
    gradienteY = convolucaoOpenCV(imagem, sobelY)
    gradienteX = np.array(gradienteX)
    gradienteY = np.array(gradienteY)

    gradiente = np.sqrt(np.power(gradienteX, 2) + np.power(gradienteY, 2))

    return gradiente

def laplacianoSum(imagem):
    laplaciano = matrizLaplaciana(3)
    laplaciano = convolucao(imagem, laplaciano)
    sum = np.array(imagem) + np.array(laplaciano)
    return Image.fromarray(sum)

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

    
    plot(imagem, convolucao(imagem, matrizMedia(5)), "Original", "Média 5x5 manual")
    plot(imagem, convolucaoSciPy(imagem, matrizGaussiana(7, 1)), "Original", "Guassiano 7x7 SciPy")
    plot(imagem, convolucaoOpenCV(imagem, matrizLaplaciana(3)), "Original", "Laplaciano 3x3 OpenCV")
    plot(imagem, convolucaoOpenCV(imagem, matrizSobelX()), "Original", "SobelX OpenCV")
    plot(imagem, convolucaoSciPy(imagem, matrizSobelY()), "Original", "SobelY Scipy")
    plot(imagem, gradiente(imagem), "Original", "Gradiente")
    plot(imagem, laplacianoSum(imagem), "Original", "Laplaciano + Original")
    
if __name__ == "__main__":
    main()
