import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy

def plot(img, edit, txt1, txt2):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(txt1)
    ax[1].imshow(edit, cmap='gray')
    ax[1].set_title(txt2)
    plt.show()

def plot3(img1, img2, img3, txt1, txt2, txt3):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(img1, cmap='gray')
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

def logaritmica(imagem, contraste, brilho):
    npImg = np.array(imagem)
    c = contraste
    img = c * np.log(1 + npImg) + brilho
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img

def potencia(imagem, c, gama):
    npImg = np.array(imagem)
    img = c * np.power(npImg, gama)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img

def negativo(npImg):
    return 255 - npImg

def medianaScipy(npImg):
    r = 3
    return Image.fromarray(scipy.ndimage.median_filter(npImg, size=r))

def planoDeBits(imagem):
    img = np.array(imagem)
    bit_planes = [(img >> i) & 1 for i in range(8)]
    
    fig, ax = plt.subplots(nrows=3, ncols=3)
    ax[0,0].imshow(bit_planes[0], cmap='gray')
    ax[0,0].set_title("Bit 0")
    ax[0,1].imshow(bit_planes[1], cmap='gray')
    ax[0,1].set_title("Bit 1")
    ax[0,2].imshow(bit_planes[2], cmap='gray')
    ax[0,2].set_title("Bit 2")
    ax[1,0].imshow(bit_planes[3], cmap='gray')
    ax[1,0].set_title("Bit 3")
    ax[1,1].imshow(bit_planes[4], cmap='gray')
    ax[1,1].set_title("Bit 4")
    ax[1,2].imshow(bit_planes[5], cmap='gray')
    ax[1,2].set_title("Bit 5")
    ax[2,0].imshow(bit_planes[6], cmap='gray')
    ax[2,0].set_title("Bit 6")
    ax[2,1].imshow(bit_planes[7], cmap='gray')
    ax[2,1].set_title("Bit 7")
    ax[2,2].imshow(imagem, cmap='gray')
    ax[2,2].set_title("Original")
    plt.show()

def outra():
    print("Digite o nome da imagem: ")
    entrada = input()
    imagem = Image.open(f'imgs/{entrada}')

    print("Escolha a transformação: ")
    print("1 - Logarítmica")
    print("2 - Potência")

    entrada = input()

    match entrada:
        case "1":
            contraste = int(input("Digite o contraste: "))
            brilho = int(input("Digite o brilho: "))
            log = logaritmica(np.array(imagem), contraste, brilho)
            plot(imagem, log, "Original", "Transformação Logarítmica")
        case "2":
            c = int(input("Digite o valor de c: "))
            gama = float(input("Digite o valor de gama: "))
            pot = potencia(np.array(imagem), c, gama)
            plot(imagem, pot, "Original", "Transformação Potência")
        case _:
            print("Opção inválida")
            exit()
    exit()

def main():
    fig38 = Image.open('imgs/Fig0308(a)(fractured_spine).tif')
    enhanceMe = Image.open('imgs/enhance-me.gif')
    imagem = None
    log = None
    pot = None
    enhanceMe = enhanceMe.convert('RGB')

    print("Escolha a imagem: ")
    print("1 - Fig 3.8 (fractured_spine)")
    print("2 - Enhance me")
    print("3 - Outra")
    entrada = input()
    
    match entrada:
        case "1":
            imagem = fig38

            log = logaritmica(np.array(imagem), 30, 20)

            pot = potencia(np.array(imagem), 1, 0.45)

        case "2":
            imagem = enhanceMe

            log = medianaScipy(np.array(imagem))
            log = logaritmica(log, 50, -20)
            log = medianaScipy(np.array(log))

            pot = medianaScipy(np.array(imagem))
            pot = potencia(np.array(pot), 1.75, 1.55)
            pot = medianaScipy(np.array(pot))
        case "3":
            outra()

        case _:
            print("Opção inválida")
            exit()

    
    plot(imagem, log, "Original", "Transformação Logarítmica")
    plot(imagem, pot, "Original", "Transformação Potência")

    planoDeBits(imagem.convert('L'))

if __name__ == "__main__":
    main()