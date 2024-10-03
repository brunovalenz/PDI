# Exemplo de leitura e plot de imagens
# Bibliotecas Numpy, Pillow, Matplotlib

import numpy as np
from PIL import Image # pillow
import matplotlib.pyplot as plt
import scipy.ndimage

# OPERAÇÃO PONTO A PONTO
def negativo(npImg):
    return 255 - npImg

def intensidade(npImg):
    return npImg/2

def qPreto(npImg, size):
    if(size%2 == 0):
      size -= 1
    half = int((size-1)/2)

    pix = 150

    if(pix%2 == 0):
      pix -= 1

    phalf = int((pix-1)/2)

    img = npImg.copy()

    img[half-phalf:half+phalf, half-phalf:half+phalf] = 0

    return img

def qBranco(npImg, size):
    pix = 100
    img = npImg.copy()
    img[0:pix, 0:pix] = 255
    img[size-pix:size, 0:pix] = 255
    img[0:pix, size-pix:size] = 255
    img[size-pix:size, size-pix:size] = 255
    return img

# OPERAÇÃO POR VIZINHANÇA

def media(xo, yo, npImg):
    r = 3
    half = int((r-1)/2)
    res = npImg.copy()

    for x in range(1, xo-1):
      for y in range(1, yo-1):
        res[x,y] = int( np.sum(npImg[x-half:x+half+1,y-half:y+half+1]) / (r*r) )
    
    return res

def mediana(xo, yo, npImg):
    r = 3
    half = int((r-1)/2)
    res = npImg.copy()
    phalf = int(((r*r)-1)/2)

    for x in range(1, xo-1):
      for y in range(1, yo-1):
        res[x,y] = np.sort(npImg[x-half:x+half+1,y-half:y+half+1], axis=None)[phalf]
    
    return res

# TRANSFORMAÇÕES GEOMÉTRICAS

def escala(npImg, escala):
    return scipy.ndimage.zoom(npImg, escala)

def rotacionar(npImg, grau):
   return scipy.ndimage.rotate(npImg, grau, reshape=True)

def translacao(npImg, x, y):
    return scipy.ndimage.shift(npImg, shift=(x, y), mode='constant', cval=0)

def plot(img, edit, txt):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Imagem original")
    ax[1].imshow(edit, cmap='gray')
    ax[1].set_title(txt)
    plt.show()


def main():

    img1 = Image.open('imgs/lena_gray_512.tif')
    img2 = Image.open('imgs/house.tif')
    img3 = Image.open('imgs/cameraman.tif')
    img4 = Image.open('imgs/lena_gray_512_salt_pepper.tif')

    # converte imagens para numpy array
    npImg1 = np.array(img1)
    npImg2 = np.array(img2)
    npImg3 = np.array(img3)
    npImg4 = np.array(img4)

    # OPERAÇÃO PONTO A PONTO

    # Negativo
    plot(img1, Image.fromarray(negativo(npImg1)), "Negativo")
    plot(img2, Image.fromarray(negativo(npImg2)), "Negativo")
    plot(img3, Image.fromarray(negativo(npImg3)), "Negativo")

    # Intensidade
    plot(img1, Image.fromarray(intensidade(npImg1)), "Intensidade")
    plot(img2, Image.fromarray(intensidade(npImg2)), "Intensidade")
    plot(img3, Image.fromarray(intensidade(npImg3)), "Intensidade")

    # Quadrados Brancos
    plot(img1, Image.fromarray(qBranco(npImg1, img1.size[0])), "Quadrados Brancos")
    plot(img2, Image.fromarray(qBranco(npImg2, img2.size[0])), "Quadrados Brancos")
    plot(img3, Image.fromarray(qBranco(npImg3, img3.size[0])), "Quadrados Brancos")

    # Quadrados Pretos
    plot(img1, Image.fromarray(qPreto(npImg1, img1.size[0])), "Quadrados Pretos")
    plot(img2, Image.fromarray(qPreto(npImg2, img2.size[0])), "Quadrados Pretos")
    plot(img3, Image.fromarray(qPreto(npImg3, img3.size[0])), "Quadrados Pretos")

    # OPERAÇÃO POR VIZINHANÇA

    # filtro da média
    plot(img4, Image.fromarray(media(img4.size[0], img4.size[1], npImg4)), "filtro da média")
    plot(img2, Image.fromarray(media(img2.size[0], img2.size[1], npImg2)), "filtro da média")
    plot(img3, Image.fromarray(media(img3.size[0], img3.size[1], npImg3)), "filtro da média")
    
    # filtro da mediana
    plot(img4, Image.fromarray(mediana(img4.size[0], img4.size[1], npImg4)), "filtro da mediana")
    plot(img2, Image.fromarray(mediana(img2.size[0], img2.size[1], npImg2)), "filtro da mediana")
    plot(img3, Image.fromarray(mediana(img3.size[0], img3.size[1], npImg3)), "filtro da mediana")

    # TRANSFORMAÇÕES GEOMÉTRICAS

    # Escala Redução em 1.5x e aumentar em 2.5x
    plot(img1, Image.fromarray(escala(npImg1, (1/1.5))), "Redução em 1.5x")
    plot(img2, Image.fromarray(escala(npImg2, (1/1.5))), "Redução em 1.5x")
    plot(img3, Image.fromarray(escala(npImg3, (1/1.5))), "Redução em 1.5x")

    plot(img1, Image.fromarray(escala(npImg1, 2.5)), "Aumento em 2.5x")
    plot(img2, Image.fromarray(escala(npImg2, 2.5)), "Aumento em 2.5x")
    plot(img3, Image.fromarray(escala(npImg3, 2.5)), "Aumento em 2.5x")

    # Rotação em 45º, 90º e 100º
    plot(img1, Image.fromarray(rotacionar(npImg1, 45)), "Rotacionada 45°")
    plot(img2, Image.fromarray(rotacionar(npImg2, 45)), "Rotacionada 45°")
    plot(img3, Image.fromarray(rotacionar(npImg3, 45)), "Rotacionada 45°")

    plot(img1, Image.fromarray(rotacionar(npImg1, 90)), "Rotacionada 90°")
    plot(img2, Image.fromarray(rotacionar(npImg2, 90)), "Rotacionada 90°")
    plot(img3, Image.fromarray(rotacionar(npImg3, 90)), "Rotacionada 90°")

    plot(img1, Image.fromarray(rotacionar(npImg1, 100)), "Rotacionada 100°")
    plot(img2, Image.fromarray(rotacionar(npImg2, 100)), "Rotacionada 100°")
    plot(img3, Image.fromarray(rotacionar(npImg3, 100)), "Rotacionada 100°")

    # Translação em 35 pixel no eixo X, 45 eixo Y
    plot(img1, Image.fromarray(translacao(npImg1, 35, 45)), "Translação em 35, 45")
    plot(img2, Image.fromarray(translacao(npImg2, 35, 45)), "Translação em 35, 45")
    plot(img3, Image.fromarray(translacao(npImg3, 35, 45)), "Translação em 35, 45")

if __name__ == "__main__":
    main()