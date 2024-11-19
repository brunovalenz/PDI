import numpy as np
import matplotlib.pyplot as plt

linhas = 9
colunas = 22
image_matrix = np.zeros([linhas, colunas])
print(image_matrix.shape)

# B C R V

#B
image_matrix[1:8,1] = 255

image_matrix[1,2:4] = 255
image_matrix[2:4,4] = 255

image_matrix[4,2:4] = 255

image_matrix[7,2:4] = 255
image_matrix[5:7,4] = 255

#C
image_matrix[2:7,6] = 255

image_matrix[1,7:10] = 255
image_matrix[7,7:10] = 255

#R
image_matrix[1:8,11] = 255

image_matrix[1,12:14] = 255
image_matrix[2:4,14] = 255

image_matrix[4,12:14] = 255

image_matrix[5,13] = 255
image_matrix[6:8,14] = 255

#V
image_matrix[1:5,16] = 255
image_matrix[1:5,20] = 255

image_matrix[4:7,17] = 255
image_matrix[4:7,19] = 255

image_matrix[6:8,18] = 255


plt.imshow(image_matrix, cmap='gray')
plt.show()