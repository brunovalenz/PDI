import datetime
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2

def plot(img, edit, txt1, txt2):

    if len(np.array(img).shape) == 3 and np.array(img).shape[2] == 3:
        img = np.mean(np.array(img), axis=2)
    if len(np.array(edit).shape) == 3 and np.array(edit).shape[2] == 3:
        edit = np.mean(np.array(edit), axis=2)

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

def fft(imagem):
    img = imagem.convert("L")
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

def plot3d(imagem):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = np.array(imagem)
    e = np.log(1 + np.abs(espectro(img)))
    x = np.arange(e.shape[1])
    y = np.arange(e.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, e, cmap='viridis')
    ax.set_title('Espectro 3D')
    plt.show()

def sinc(size = 512):
    img = np.ones((size, size), dtype=np.float32) * 255  

    square_size = 50
    center = size // 2
    half_square = square_size // 2
    img[center - half_square:center + half_square, center - half_square:center + half_square] = 0

    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    sinc_func = np.sinc(np.sqrt(X**2 + Y**2))

    sinc_func_normalized = 255 * (sinc_func - sinc_func.min()) / (sinc_func.max() - sinc_func.min())

    img = img * sinc_func_normalized

    return Image.fromarray(img).convert("L")


def main():
    car = Image.open('6-Transformada de Fourier/imgs/car.tif')
    lena = Image.open('6-Transformada de Fourier/imgs/len_periodic_noise.png')
    newspaper = Image.open('6-Transformada de Fourier/imgs/newspaper_shot_woman.tif')
    indian = Image.open('6-Transformada de Fourier/imgs/periodic_noise.png')
    sincIMG = Image.open('6-Transformada de Fourier/imgs/sinc.png')
    imagem = None

    print("Escolha a imagem: ")
    print("1 - Car")
    print("2 - Lena")
    print("3 - Newspaper")
    print("4 - Indian")
    print("5 - Sinc")
    print("6 - Gerada")
    entrada = input()
    
    match entrada:
        case "1":
            imagem = car

        case "2":
            imagem = lena

        case "3":
            imagem = newspaper
        
        case "4":
            imagem = indian
        
        case "5":
            imagem = sincIMG
        
        case "6":
            imagem = sinc()

        case _:
            print("Opção inválida")
            exit()

    plot(imagem, fftImage(imagem), "Original", "FFT")

    plot(imagem, inverse_fftImage(fft(imagem)), "Original", "Depois da FFT inversa")

    plotFFT(imagem)

    plot3d(imagem)
    
if __name__ == "__main__":
    main()
