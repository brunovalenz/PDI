import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plot(img, edit, txt1, txt2):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(txt1)
    ax[1].imshow(edit, cmap='gray')
    ax[1].set_title(txt2)
    plt.show()

def codificar(imagem, texto):
    if imagem.mode != 'RGB':
        imagem = imagem.convert('RGB')
        
    img = np.array(imagem)

    texto = texto + 'EOF'

    texto_bin = ''.join(format(ord(i), '08b') for i in texto)
    
    index = 0

    for i in range(imagem.size[0]):

        for j in range(imagem.size[1]):

            for k in range(3):

                if index < len(texto_bin):

                    img[i][j][k] = int(bin(img[i][j][k])[2:9] + texto_bin[index], 2)

                    index += 1

                else:

                    break

    img = Image.fromarray(img)

    return img

def decodificar(imagem):
    if imagem.mode != 'RGB':
        imagem = imagem.convert('RGB')
        
    img = np.array(imagem)

    texto_bin = ""

    for i in range(img.shape[0]):

        for j in range(img.shape[1]):

            for k in range(3):

                texto_bin += bin(img[i, j, k])[2:].zfill(8)[-1]

    texto = ""

    for i in range(0, len(texto_bin), 8):

        texto += chr(int(texto_bin[i:i+8], 2))

        if texto[-3:] == "EOF":

            break

    return texto[:-3]

def mensagem(imagem):
    pix = imagem.size[0] * imagem.size[1]
    chars = ((pix * 3) // 8) - 3

    print("Tamanho maximo de caracters da mensagem: ", chars)
    print("Digite a mensagem: ")
    entrada = input()

    if len(entrada) > chars:
        print("Mensagem muito grande")
        return
    
    return entrada

def salvar(imagem, nome):
    print("Desenha salvar a imagem? (s/N)")
    entrada = input()
    if entrada == "s":
        time = datetime.datetime.now().strftime("%Y-%m-%d")
        imagem.save(f'imgs/{nome}{time}.png')

def main():
    jadirCinza = Image.open('imgs/NEWjadirCinza.jpg')
    jadir = Image.open('imgs/jadirs.jpg')

    print("Escolha a imagem: ")
    print("1 - Jadir Cinza")
    print("2 - Jadir")
    print("3 - Outra")
    print("4 - Decodificar Imagem")
    entrada = input()
    
    match entrada:
        case "1":
            txt = mensagem(jadirCinza)
            codificada = codificar(jadirCinza, txt)
            plot(jadirCinza, codificada, "Imagem Original", "Imagem Editada")
            salvar(codificada, "Jadir Cinza")
            print("Mensagem: ", decodificar(codificada))
        case "2":
            txt = mensagem(jadir)
            codificada = codificar(jadir, txt)
            plot(jadir, codificada, "Imagem Original", "Imagem Editada")
            salvar(codificada, "Jadir")
            print("Mensagem: ", decodificar(codificada))
        case "3":
            print("Digite o nome da imagem: ")
            entrada = input()
            imagem = Image.open("imgs/" + entrada)
            txt = mensagem(imagem)
            codificada = codificar(imagem, txt)
            plot(imagem, codificada, "Imagem Original", "Imagem Editada")
            salvar(codificada, entrada)
            print("Mensagem: ", decodificar(codificada))
        case "4":
            print("Digite o nome da imagem: ")
            entrada = input()
            imagem = Image.open("imgs/" + entrada)
            print("Mensagem: ", decodificar(imagem)) 
        


    #plot(jadir,jadirCinza, "Jadir Normal", "JADIR CINZA")


if __name__ == "__main__":
    main()
