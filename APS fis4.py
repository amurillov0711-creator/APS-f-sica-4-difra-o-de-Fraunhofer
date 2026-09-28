import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------
# ENTRADAS
# ---------------------------------------------------------

L = float(input("Distância até o anteparo [m]: "))
altura_objeto = float(input("Altura física do objeto [m]: "))
lambda_nm = float(input("Comprimento de onda [nm]: "))

caminho_imagem = input("Caminho da imagem: ")

# Converte nm -> m
lamb = lambda_nm * 1e-9

# ---------------------------------------------------------
# LEITURA DA IMAGEM
# ---------------------------------------------------------

imagem = Image.open(caminho_imagem).convert("L")

A = np.array(imagem, dtype=float)

# Normaliza a imagem entre 0 e 1
A = A / 255.0

ny, nx = A.shape

# A altura física da imagem foi fornecida pelo usuário.
# A largura física é calculada mantendo a proporção da imagem.
largura_objeto = altura_objeto * nx / ny

# Tamanho físico correspondente a cada pixel
dy = altura_objeto / ny
dx = largura_objeto / nx

print("\n--- Dados utilizados ---")
print(f"Distância até o anteparo: {L:.4e} m")
print(f"Comprimento de onda: {lamb:.4e} m")
print(f"Altura do objeto: {altura_objeto:.4e} m")
print(f"Largura do objeto: {largura_objeto:.4e} m")
print(f"Resolução da imagem: {nx} x {ny} pixels")

# ---------------------------------------------------------
# TRANSFORMADA DE FOURIER
# ---------------------------------------------------------

# FFT bidimensional da abertura
E = np.fft.fft2(A)

# Coloca a frequência zero no centro da matriz
E = np.fft.fftshift(E)

# Intensidade luminosa
I = np.abs(E) ** 2

# Normalização
I = I / np.max(I)

# ---------------------------------------------------------
# COORDENADAS ESPACIAIS NO ANTEPARO
# ---------------------------------------------------------

# Frequências espaciais da FFT
fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))

# Aproximação de Fraunhofer/paraxial:
#
# x_anteparo ≈ lambda * L * fx
# y_anteparo ≈ lambda * L * fy

x_anteparo = lamb * L * fx
y_anteparo = lamb * L * fy

# Converte para milímetros para facilitar a visualização
x_mm = x_anteparo * 1000
y_mm = y_anteparo * 1000

# ---------------------------------------------------------
# VISUALIZAÇÃO
# ---------------------------------------------------------

plt.figure(figsize=(10, 8))

# Escala logarítmica para enxergar os máximos secundários
I_visual = np.log10(1 + 1000 * I)

plt.imshow(
    I_visual,
    extent=[
        x_mm[0],
        x_mm[-1],
        y_mm[0],
        y_mm[-1]
    ],
    origin="lower",
    cmap="inferno",
    aspect="equal"
)

plt.colorbar(label="Intensidade relativa (escala logarítmica)")

plt.xlabel("x no anteparo [mm]")
plt.ylabel("y no anteparo [mm]")

plt.title(
    f"Figura de Difração\n"
    f"λ = {lambda_nm:.1f} nm | L = {L:.2f} m"
)

plt.tight_layout()

plt.show()