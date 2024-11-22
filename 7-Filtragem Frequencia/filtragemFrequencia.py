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

def plotDois(imgs):
    for i in range(len(imgs)):
        fig, ax = plt.subplots(nrows=2, ncols=4)
        ax[0][0].imshow(imgs[i], cmap='gray')
        ax[0][0].set_title("Original")

        ax[0][1].imshow(fftImage(imgs[i]), cmap='gray')
        ax[0][1].set_title("Espectro")

        ax[0][2].imshow(imgFiltro(filtroPassaBaixa(20, imgs[i].shape, tipo="Ideal", porcentagem=True)), cmap='gray')
        ax[0][2].set_title("Filtro Ideal - 20%")

        ax[0][3].imshow(Filtrar(imgs[i], filtroPassaBaixa(20, imgs[i].shape, tipo="Ideal", porcentagem=True)), cmap='gray')
        ax[0][3].set_title("Filtrada Ideal - 20%")

        ax[1][0].imshow(imgFiltro(filtroPassaBaixa(20, imgs[i].shape, tipo="Butterworth", porcentagem=True)), cmap='gray')
        ax[1][0].set_title("Filtro Butterworth - 20%")

        ax[1][1].imshow(Filtrar(imgs[i], filtroPassaBaixa(20, imgs[i].shape, tipo="Butterworth", porcentagem=True)), cmap='gray')
        ax[1][1].set_title("Filtrada Butterworth - 20%")

        ax[1][2].imshow(imgFiltro(filtroPassaBaixa(20, imgs[i].shape, tipo="Gaussiano", porcentagem=True)), cmap='gray')
        ax[1][2].set_title("Filtro Gaussiano - 20%")

        ax[1][3].imshow(Filtrar(imgs[i], filtroPassaBaixa(20, imgs[i].shape, tipo="Gaussiano", porcentagem=True)), cmap='gray')
        ax[1][3].set_title("Filtrada Gaussiano - 20%")
        plt.show()

def plotTres(imgs):
    for i in range(len(imgs)):
        fig, ax = plt.subplots(nrows=2, ncols=4)
        ax[0][0].imshow(imgs[i], cmap='gray')
        ax[0][0].set_title("Original")

        ax[0][1].imshow(fftImage(imgs[i]), cmap='gray')
        ax[0][1].set_title("Espectro")

        ax[0][2].imshow(imgFiltro(filtroPassaAlta(20, imgs[i].shape, tipo="Ideal", porcentagem=True)), cmap='gray')
        ax[0][2].set_title("Filtro Ideal - 20%")

        ax[0][3].imshow(Filtrar(imgs[i], filtroPassaAlta(20, imgs[i].shape, tipo="Ideal", porcentagem=True)), cmap='gray')
        ax[0][3].set_title("Filtrada Ideal - 20%")

        ax[1][0].imshow(imgFiltro(filtroPassaAlta(20, imgs[i].shape, tipo="Butterworth", porcentagem=True)), cmap='gray')
        ax[1][0].set_title("Filtro Butterworth - 20%")

        ax[1][1].imshow(Filtrar(imgs[i], filtroPassaAlta(20, imgs[i].shape, tipo="Butterworth", porcentagem=True)), cmap='gray')
        ax[1][1].set_title("Filtrada Butterworth - 20%")

        ax[1][2].imshow(imgFiltro(filtroPassaAlta(20, imgs[i].shape, tipo="Gaussiano", porcentagem=True)), cmap='gray')
        ax[1][2].set_title("Filtro Gaussiano - 20%")

        ax[1][3].imshow(Filtrar(imgs[i], filtroPassaAlta(20, imgs[i].shape, tipo="Gaussiano", porcentagem=True)), cmap='gray')
        ax[1][3].set_title("Filtrada Gaussiano - 20%")
        plt.show()


def salvar(imagem, nome):
    print("Desenha salvar a imagem? (s/N)")
    entrada = input()
    if entrada == "s":
        time = datetime.datetime.now().strftime("%Y-%m-%d")
        imagem.save(f'imgs/{nome}{time}.png')

def baseImage(imgSize, blackSize, percentage = False):
    img = np.zeros((imgSize, imgSize))
    center = imgSize // 2
    if percentage:
        blackSize = int(imgSize * (blackSize / 100))
    img[center - blackSize // 2:center + blackSize // 2, center - blackSize // 2:center + blackSize // 2] = 1
    return img

def fft(imagem):
    
    if not isinstance(imagem, np.ndarray):
        img = imagem.convert("L")
    else:
        img = imagem
    img = np.array(imagem)
    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)
    return fft

def fftImage(imagem):
    transf = fft(imagem)
    transf = np.log(1 + np.abs(transf))
    transf = 255 * (transf - np.min(transf)) / (np.max(transf) - np.min(transf))

    return Image.fromarray(np.uint8(transf))

def inverse_fft(fft):
    ifft = np.fft.ifftshift(fft)
    ifft = np.fft.ifft2(ifft)
    return np.real(ifft)

def inverse_fftImage(fft):
    ifft = inverse_fft(fft)
    ifft = 255 * (ifft - np.min(ifft)) / (np.max(ifft) - np.min(ifft))

    return Image.fromarray(np.uint8(ifft))

def espectro(imagem):
    img = np.array(imagem)
    return np.fft.fftshift(np.fft.fft2(img))

def fase(imagem):
    img = np.array(imagem)
    return np.angle(np.fft.fftshift(np.fft.fft2(img)))

def plotFFT(imagem):
    e = espectro(imagem)
    f = fase(imagem)
    plot(np.log(1 + np.abs(e)), f, "Espectro", "Fase")

def rotacionar(imagem, angulo):
    imagem = Image.fromarray(np.array(imagem))
    imagem = imagem.rotate(angulo)
    return np.array(imagem)

def transladar(imagem, dx, dy):
    imagem = Image.fromarray(np.array(imagem))
    imagem = imagem.transform(imagem.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
    return np.array(imagem)

def zoom(imagem, fator):
    imagem = Image.fromarray(np.array(imagem))
    imagem = imagem.resize((int(imagem.width * fator), int(imagem.height * fator)))
    return np.array(imagem)

def h():
    print("h) ")
    print("Rotação: Uma rotação no espaço resulta em uma rotação idêntica no domínio da frequência.")
    print("Translação: Um deslocamento da imagem no espaço altera apenas sua fase no domínio da frequência.")
    print("Zoom: Ampliar a imagem no espaço reduz sua transformada de Fourier.")

import numpy as np

def filtroPassaBaixa(raio, imgSize, tipo="Ideal", ordem=2, porcentagem=False):
    altura, largura = imgSize
    filtro = np.zeros((altura, largura))
    
    centro_x, centro_y = altura // 2, largura // 2
    
    if porcentagem:
        diagonal = np.sqrt(altura**2 + largura**2)
        raio = (raio / 100) * (diagonal / 2)

    x, y = np.meshgrid(np.arange(largura), np.arange(altura))
    dist = np.sqrt((x - centro_y)**2 + (y - centro_x)**2)
    
    match tipo:
        case "Ideal":
            filtro[dist <= raio] = 1
        case "Butterworth":
            filtro = 1 / (1 + (dist / raio)**(2 * ordem))
        case "Gaussiano":
            filtro = np.exp(-(dist**2) / (2 * raio**2))
        case _:
            raise ValueError("Tipo de filtro inválido")
    
    return filtro

def filtroPassaAlta(raio, imgSize, tipo="Ideal", ordem=2, porcentagem=False):
    return 1 - filtroPassaBaixa(raio, imgSize, tipo, ordem, porcentagem)

def filtroPassaFaixa(raio1, raio2, imgSize, tipo="Ideal", ordem=2, porcentagem=False):
    return filtroPassaBaixa(raio2, imgSize, tipo, ordem, porcentagem) - filtroPassaBaixa(raio1, imgSize, tipo, ordem, porcentagem)

def Filtrar(imagem, filtro):
    filtro = fft(imagem) * filtro
    return inverse_fft(filtro)

def imgFiltro(filtro):
    filtro = 255 * (filtro - np.min(filtro)) / (np.max(filtro) - np.min(filtro))
    return np.uint8(filtro)

def loadFourierImgs():
    path = '7-Filtragem Frequencia/imgs/fourier/'
    imgs = []
    imgs.append(np.array(Image.open(path + 'car.tif').convert('L')))
    imgs.append(np.array(Image.open(path + 'len_periodic_noise.png').convert('L')))
    imgs.append(np.array(Image.open(path + 'newspaper_shot_woman.tif').convert('L')))
    imgs.append(np.array(Image.open(path + 'periodic_noise.png').convert('L')))
    imgs.append(np.array(Image.open(path + 'pnois2.jpg').convert('L')))
    return imgs

def ExUm():

    imagem = baseImage(512, 25, True)
    plotFFT(imagem)
    rotacionada = rotacionar(imagem, 40)
    plot(imagem, rotacionada, "Original", "Rotacionada 40°")
    plotFFT(rotacionada)
    transladada = transladar(imagem, 50, 50)
    plot(imagem, transladada, "Original", "Transladada 50")
    plotFFT(transladada)
    zoomIMG = zoom(imagem, 0.75)
    plot(imagem, zoomIMG, "Original", "Zoom")
    plotFFT(zoomIMG)
    h()

def ExDois():
    imgs = loadFourierImgs()
    plotDois(imgs)

def ExTres():
    imgs = loadFourierImgs()
    plotTres(imgs)

def ExQuatro():
    imgs = loadFourierImgs()
    imgs[0] = Filtrar(imgs[0], filtroPassaBaixa(20, imgs[0].shape, tipo="Ideal", porcentagem=True))
    imgs[1] = Filtrar(imgs[1], filtroPassaBaixa(20, imgs[1].shape, tipo="Ideal", porcentagem=True)) 
    imgs[2] = Filtrar(imgs[2], filtroPassaBaixa(10, imgs[2].shape, tipo="Ideal", porcentagem=True))
    imgs[3] = Filtrar(imgs[3], filtroPassaBaixa(20, imgs[3].shape, tipo="Ideal", porcentagem=True))
    imgs[4] = Filtrar(imgs[4], filtroPassaBaixa(2, imgs[4].shape, tipo="Butterworth", porcentagem=True))
    for img in imgs:
        img = Image.fromarray(img)
        img.show()

def ExCinco():
    imgs = loadFourierImgs()
    imgs[0] = Filtrar(imgs[0], filtroPassaAlta(20, imgs[0].shape, tipo="Ideal", porcentagem=True))
    imgs[1] = Filtrar(imgs[1], filtroPassaAlta(20, imgs[1].shape, tipo="Ideal", porcentagem=True)) 
    imgs[2] = Filtrar(imgs[2], filtroPassaAlta(10, imgs[2].shape, tipo="Ideal", porcentagem=True))
    imgs[3] = Filtrar(imgs[3], filtroPassaAlta(20, imgs[3].shape, tipo="Ideal", porcentagem=True))
    imgs[4] = Filtrar(imgs[4], filtroPassaAlta(4, imgs[4].shape, tipo="Butterworth", porcentagem=True))
    for img in imgs:
        img = Image.fromarray(img)
        img.show()

def ExSeis():
    imgs = loadFourierImgs()
    pos = 4
    img = imgs[pos]
    filtro = filtroPassaFaixa(5, 30, img.shape, tipo="Ideal", porcentagem=True)
    filtroIMG = imgFiltro(filtro)
    img = Filtrar(img, filtro)
    plot(filtroIMG, img, "Filtro", "Imagem filtrada")
    print("O filtro passa-faixa é um tipo de filtro que permite a passagem de frequências dentro de uma faixa específica bloqueia as frequências fora dessa faixa.")
    print("Combinação de um filtro passa-alta e um filtro passa-baixa.")

def main():

    ExUm()
    # h) 
    # Rotação: Uma rotação no espaço resulta em uma rotação idêntica no domínio da frequência.
    # Translação: Um deslocamento da imagem no espaço altera apenas sua fase no domínio da frequência.
    # Zoom: Ampliar a imagem no espaço reduz sua transformada de Fourier.
    ExDois()
    ExTres()
    ExQuatro()
    ExCinco()
    ExSeis()
    # O filtro passa-faixa é um tipo de filtro que permite a passagem de frequências dentro de uma faixa específica bloqueia as frequências fora dessa faixa.
    # Combinação de um filtro passa-alta e um filtro passa-baixa.
    
if __name__ == "__main__":
    main()
