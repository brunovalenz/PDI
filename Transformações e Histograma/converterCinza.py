from PIL import Image

def rgb_to_grayscale(image_path, output_path):
    # Carrega a imagem RGB
    imagem_rgb = Image.open(image_path)

    # Converte a imagem para grayscale
    imagem_gray = imagem_rgb.convert('L')

    # Salva a imagem em escala de cinza
    imagem_gray.save(output_path)
    print(f"Imagem convertida para grayscale e salva em: {output_path}")

# Exemplo de uso
image_path = 'imgs/jadirs.jpg'  # Substitua pelo caminho da sua imagem
output_path = 'imgs/NEWjadirCinza.jpg'  # Substitua pelo caminho onde deseja salvar a imagem
rgb_to_grayscale(image_path, output_path)
