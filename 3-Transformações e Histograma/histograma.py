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

def plotHistograma(img, opcao):
    img_array = np.array(img)
    if opcao == 1:
        normalized = normalizarHistograma(img)
    else :
        normalized = normalizarHistogramaOpenCV(img)

    hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    hist_normalized = cv2.calcHist([np.array(normalized)], [0], None, [256], [0, 256])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    ax[0, 0].imshow(img, cmap='gray', aspect='auto')
    ax[0, 0].set_title('Imagem Original')
    ax[0, 0].axis('off')

    ax[0, 1].imshow(normalized, cmap='gray', aspect='auto')
    if opcao == 1:
        ax[0, 1].set_title('Imagem com Histograma Normalizado usando Numpy')
    else:
        ax[0, 1].set_title('Imagem com Histograma Normalizado usando OpenCV')

    ax[0, 1].set_title('Imagem com Histograma Normalizado')
    ax[0, 1].axis('off')

    ax[1, 0].plot(hist, color='gray')
    ax[1, 0].set_title('Histograma Original')
    ax[1, 0].set_xlim([0, 256])
    ax[1, 0].set_ylim([0, max(hist) * 1.1])

    ax[1, 1].plot(hist_normalized, color='gray')
    ax[1, 1].set_title('Histograma Normalizado')
    ax[1, 1].set_xlim([0, 256])
    ax[1, 1].set_ylim([0, max(hist_normalized) * 1.1])

    fig.tight_layout()

    
    plt.show()

def salvar(imagem, nome):
    print("Desenha salvar a imagem? (s/N)")
    entrada = input()
    if entrada == "s":
        time = datetime.datetime.now().strftime("%Y-%m-%d")
        imagem.save(f'imgs/{nome}{time}.png')

def normalizarHistogramaOpenCV(imagem):
    img = np.array(imagem)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / (img.shape[0] * img.shape[1])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255
    cdf_normalized = cdf_normalized.astype('uint8')
    img_normalized = cdf_normalized[img]
    return Image.fromarray(img_normalized)

def normalizarHistograma(imagem):
    img = np.array(imagem)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')
    img_normalized = cdf_normalized[img]
    return Image.fromarray(img_normalized)

def main():
    fig38 = Image.open('imgs/Fig0308(a)(fractured_spine).tif')
    enhanceMe = Image.open('imgs/enhance-me.gif')
    jadir = Image.open('imgs/jadirs.jpg')
    imagem = None
    enhanceMe = enhanceMe.convert('RGB')

    print("Escolha a imagem: ")
    print("1 - Fig 3.8 (fractured_spine)")
    print("2 - Enhance me")
    print("3 - Jadir")
    entrada = input()
    
    match entrada:
        case "1":
            imagem = fig38


        case "2":
            imagem = enhanceMe

        case "3":
            imagem = jadir

        case _:
            print("Opção inválida")
            exit()

    plotHistograma(imagem, 1)
    plotHistograma(imagem, 2)

if __name__ == "__main__":
    main()
