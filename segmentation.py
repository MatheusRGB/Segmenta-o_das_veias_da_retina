import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Funções de métricas
def calcular_mse(imagem1, imagem2):
    erro = np.sum((imagem1 - imagem2) ** 2)
    mse = erro / float(imagem1.shape[0] * imagem1.shape[1])
    return mse

def calcular_ssim(imagem1, imagem2):
    ssim_value, _ = ssim(imagem1, imagem2, full=True)
    return ssim_value

def calcular_jaccard(imagem1, imagem2):
    intersection = np.sum(np.logical_and(imagem1, imagem2))
    union = np.sum(np.logical_or(imagem1, imagem2))
    return intersection / float(union)

def calcular_dice(imagem1, imagem2):
    intersection = np.sum(np.logical_and(imagem1, imagem2))
    dice_index = (2 * intersection) / (np.sum(imagem1) + np.sum(imagem2))
    return dice_index

def calcular_acuracia(imagem1, imagem2):
    pixels_corretos = np.sum(imagem1 == imagem2)
    total_pixels = imagem1.size
    return pixels_corretos / total_pixels

def calcular_sensibilidade(mask_seg, mask_ref):
    mask_seg_bin = np.array(mask_seg > 0, dtype=np.uint8)
    mask_ref_bin = np.array(mask_ref > 0, dtype=np.uint8)

    TP = np.sum((mask_seg_bin == 1) & (mask_ref_bin == 1))

    FN = np.sum((mask_seg_bin == 0) & (mask_ref_bin == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    return sensitivity

def calcular_especificidade(mask_seg, mask_ref):
    mask_seg_bin = np.array(mask_seg > 0, dtype=np.uint8)
    mask_ref_bin = np.array(mask_ref > 0, dtype=np.uint8)

    TN = np.sum((mask_seg_bin == 0) & (mask_ref_bin == 0))

    FP = np.sum((mask_seg_bin == 1) & (mask_ref_bin == 0))

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return specificity

kernel_pontos = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)

kernel_pontos_2 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]], dtype=np.uint8)

kernel_pontos_3 = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.uint8)

kernel_reta = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]], dtype=np.uint8)

kernel_reta2 = np.array([[0, 1, 0],
                          [0, 1, 0],
                          [0, 1, 0]], dtype=np.uint8)

kernel_diagonal = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0]], dtype=np.uint8)

kernel_diagonal2 = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.uint8)

def contornos_conectados(imagem_binaria, limite_area_pequeno, limite_area_grande):

    contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imagem_pequena = np.zeros_like(imagem_binaria)
    imagem_grande = np.zeros_like(imagem_binaria)

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if area < limite_area_pequeno:
            cv2.drawContours(imagem_pequena, [contorno], -1, 255, thickness=cv2.FILLED)
        elif area > limite_area_grande:
            cv2.drawContours(imagem_grande, [contorno], -1, 255, thickness=cv2.FILLED)

    return imagem_pequena, imagem_grande

def segmentation(image, imagem_original, mask):

  r, g, b = cv2.split(image)

  imagem = np.zeros_like(image)
  imagem = g

  imagem_complementar = cv2.bitwise_not(imagem)

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))

  imagem = clahe.apply(imagem)

  imagem_complementar = clahe.apply(imagem_complementar)


  #Elemento estruturante
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

  #Aplicação na imagem
  abertura = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)

  abertura_invertida = cv2.morphologyEx(imagem_complementar, cv2.MORPH_OPEN, kernel)

  clahe.apply(abertura)

  #Subtração do fundo da imagem
  imagem_ruidos = cv2.subtract(imagem, abertura)

  imagem_invertida_ruidos_1 = cv2.subtract(imagem_complementar, abertura_invertida)

  imagem_invertida_ruidos = np.clip(imagem_invertida_ruidos_1.astype(int) - imagem_ruidos.astype(int), 0, 255).astype(np.uint8)

  imagem_invertida_ruidos[imagem_invertida_ruidos <= 10] = 0

  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(imagem_invertida_ruidos, connectivity=8)

  min_size = 15
  ruidos_menores = np.zeros_like(imagem_invertida_ruidos)

  for i in range(1, num_labels):
      if stats[i, cv2.CC_STAT_AREA] <= min_size:
          ruidos_menores[labels == i] = 255

  imagem_invertida_ruidos = np.clip(imagem_invertida_ruidos.astype(int) - ruidos_menores.astype(int), 0, 255).astype(np.uint8)


  imagem_suave = cv2.bilateralFilter(imagem_invertida_ruidos, d=5, sigmaColor=15, sigmaSpace=4)
  imagem_suave[imagem_suave <= 10] = 0

  imagem_suave[imagem_suave >= 18] = 255

  imagem_binaria = imagem_suave

  imagem_segmentada = imagem_binaria

  imagem_segmentada = cv2.dilate(imagem_segmentada, kernel_reta, iterations=1)
  imagem_segmentada = cv2.erode(imagem_segmentada, kernel_reta, iterations=1)

  imagem_segmentada = cv2.dilate(imagem_segmentada, kernel_reta2, iterations=1)
  imagem_segmentada = cv2.erode(imagem_segmentada, kernel_reta2, iterations=1)

  imagem_segmentada = cv2.dilate(imagem_segmentada, kernel_diagonal, iterations=1)
  imagem_segmentada = cv2.erode(imagem_segmentada, kernel_diagonal, iterations=1)

  imagem_segmentada = cv2.dilate(imagem_segmentada, kernel_diagonal2, iterations=1)
  imagem_segmentada = cv2.erode(imagem_segmentada, kernel_diagonal2, iterations=1)

  imagem_pequena, imagem_grande = contornos_conectados(imagem_segmentada, 5, 1500)


  #Componentes menores
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(imagem_segmentada, connectivity=8)

  min_size = 25
  componentes_menores = np.zeros_like(imagem_segmentada)

  for i in range(1, num_labels):
      if stats[i, cv2.CC_STAT_AREA] <= min_size:
          componentes_menores[labels == i] = 255

  #Componentes maiores
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(imagem_segmentada, connectivity=8)

  min_size_big = 90
  componentes_maiores = np.zeros_like(imagem_segmentada)

  for i in range(1, num_labels):
      if stats[i, cv2.CC_STAT_AREA] >= min_size_big:
          componentes_maiores[labels == i] = 255

  imagem_segmentada = np.clip(imagem_segmentada.astype(int) - imagem_pequena.astype(int), 0, 255).astype(np.uint8)
  imagem_segmentada = np.clip(imagem_segmentada.astype(int) - componentes_menores.astype(int), 0, 255).astype(np.uint8)
  imagem_segmentada = cv2.bitwise_and(componentes_maiores, imagem_segmentada)

  imagem_final = cv2.bitwise_and(imagem_original, imagem_segmentada)


  plt.figure(figsize=(15, 5))

  plt.subplot(1, 4, 1)
  plt.imshow(imagem_original, cmap='gray')
  plt.title('Imagem Original')
  plt.axis('off')

  plt.subplot(1, 4, 2)
  plt.imshow(imagem_segmentada, cmap='gray')
  plt.title('Mascara obtida')
  plt.axis('off')

  plt.subplot(1, 4, 3)
  plt.imshow(mask, cmap='gray')
  plt.title('Mascara amostra')
  plt.axis('off')

  plt.subplot(1, 4, 4)
  plt.imshow(imagem_final, cmap='gray')
  plt.title('Imagem final')
  plt.axis('off')

  plt.tight_layout()
  plt.show()


  #MSE
  mse = calcular_mse(imagem_segmentada, mask)
  print(f"MSE: {mse}")

  #SSIM
  ssim_value = calcular_ssim(imagem_segmentada, mask)
  print(f"SSIM: {ssim_value}")

  #Jaccard
  jaccard_index = calcular_jaccard(imagem_segmentada, mask)
  print(f"Índice de Jaccard: {jaccard_index}")

  #Dice
  dice_index = calcular_dice(imagem_segmentada, mask)
  print(f"Coeficiente de Dice: {dice_index}")

  #Acuracia
  acuracia = calcular_acuracia(imagem_segmentada, mask)
  print(f"Acurácia: {acuracia}")

  #Sensibilidade
  sensibilidade = calcular_sensibilidade(imagem_segmentada, mask)
  print(f"Sensibilidade: {sensibilidade}")

  #Especificidade
  especificidade = calcular_especificidade(imagem_segmentada, mask)
  print(f"Especificidade: {especificidade}")

  return {
      "MSE": mse,
      "SSIM": ssim_value,
      "Jaccard": jaccard_index,
      "Dice": dice_index,
      "Acuracia": acuracia,
      "Sensibilidade": sensibilidade,
      "Especificidade": especificidade
    }

resultados = {
    "MSE": 0,
    "SSIM": 0,
    "Jaccard": 0,
    "Dice": 0,
    "Acuracia": 0,
    "Sensibilidade": 0,
    "Especificidade": 0
}

cont = 0

import cv2
import os
import matplotlib.pyplot as plt


pasta_imagens = "./images/"
pasta_masks = "./masks/"


nomes_imagens = sorted(os.listdir(pasta_imagens))
nomes_masks = sorted(os.listdir(pasta_masks))


if len(nomes_imagens) != len(nomes_masks):
    print("Número de imagens e máscaras não corresponde!")
else:
    for nome in nomes_imagens:
        if nome in nomes_masks:  
            
            caminho_imagem = os.path.join(pasta_imagens, nome)
            caminho_mask = os.path.join(pasta_masks, nome)

      
            imagem = cv2.imread(caminho_imagem)[:, :, ::-1]  
            imagem_original = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(caminho_mask, cv2.IMREAD_GRAYSCALE)

            metricas = segmentation(imagem, imagem_original, mask)

            for i in resultados:
              resultados[i] += metricas[i]

            cont += 1

        else:
            print(f"Arquivo {nome} não encontrado em ambas as pastas.")


medias = {chave: valor / cont for chave, valor in resultados.items()}


print("\nMedia dos resultados:")
for j, valor in medias.items():
    print(f"{j}: {valor:.4f}")
