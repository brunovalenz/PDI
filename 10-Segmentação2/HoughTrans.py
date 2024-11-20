import datetime
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
import math

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

def linhasPadrao(imagem, limiar=50, limiar2=200):
    imagem = np.array(imagem)
    imagem = cv2.GaussianBlur(imagem, (5, 5), 0)
    imagem = cv2.Canny(imagem, limiar, limiar2, apertureSize=3)
    
    linhas = cv2.HoughLines(imagem, 1, np.pi/180, 150, None, 0, 0)

    imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    if linhas is not None:
        for i in range(0, len(linhas)):
            rho = linhas[i][0][0]
            theta = linhas[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(imagem, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)  

        return imagem
    
def linhasProb(imagem, limiar=50, limiar2=200):
    imagem = np.array(imagem)
    imagem = cv2.GaussianBlur(imagem, (5, 5), 0)
    imagem = cv2.Canny(imagem, limiar, limiar2, apertureSize=3)

    linhas = cv2.HoughLinesP(imagem, 1, np.pi / 180, 50, 100, minLineLength=10, maxLineGap=250)

    imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    if linhas is not None:
      for i in range(0, len(linhas)):
          l = linhas[i][0]
          cv2.line(imagem, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)  
    
    return imagem

def circulos(imagem, minDist=50, param1=30, param2=50, minRadius=0, maxRadius=0, blur=7):
    imagem = imagem.convert('L')
    imagem = np.array(imagem)
    white = np.full(imagem.shape, 255)
    blur = cv2.GaussianBlur(imagem, (blur, blur), 0)
    plot(imagem, blur, "Original", "Blur")
    circulos = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for i in circulos[0, :]:
            cv2.circle(white, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(white, (i[0], i[1]), 2, (0, 0, 255), 3)

    return white

def main():
    papel = Image.open('Segmentação2/imgs/paper.png')
    aqua = Image.open('Segmentação2/imgs/aqua.jpg')
    som = Image.open('Segmentação2/imgs/som.jpg')
    circulo1 = Image.open('Segmentação2/imgs/circles.jpg')
    circulo2 = Image.open('Segmentação2/imgs/circles.png')
    circulo3 = Image.open('Segmentação2/imgs/circles2.jpg')
    moedas = Image.open('Segmentação2/imgs/coins.jpg')



    plot(papel, linhasPadrao(papel), "Original", "Linhas")
    plot(papel, linhasProb(papel), "Original", "Linhas")

    plot(aqua, linhasPadrao(aqua), "Original", "Linhas")
    plot(aqua, linhasProb(aqua), "Original", "Linhas")

    plot(som, linhasPadrao(som), "Original", "Linhas")
    plot(som, linhasProb(som), "Original", "Linhas")
    
    #---------------------------------------------------------------------------------- #

    plot(circulo1, circulos(circulo1), "Original", "Circulos")
    plot(circulo2, circulos(circulo2), "Original", "Circulos")
    plot(circulo3, circulos(circulo3, minDist=0.001, param1=695, param2=138.5, minRadius=0, maxRadius=0), "Original", "Circulos")
    plot(moedas, circulos(moedas, minDist=100, param1=30, param2=80, minRadius=0, maxRadius=0, blur=35), "Original", "Circulos")

    #---------------------------------------------------------------------------------- #

    

if __name__ == "__main__":
    main()
