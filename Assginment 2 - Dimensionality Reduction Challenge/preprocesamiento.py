import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from sklearn.preprocessing import StandardScaler
from crea_tabla import *

def ordenar_datos(images_data, original_data, norm=True):
    '''
    Función para importar datos y agruparlos en diccionarios


    Parameters:
    -----------
    images_data : DataFrameGroupBy
        Agrupación de tablas de características de imágenes. Cada grupo son las características de una imagen.
    original_data : array    
        Contenedor de las imágenes originales
    norm : bool    
        Determina si se normalizan los datos de las imagenes o no

    Return:
    -------
    tuple (dict, dict)    
        Diccionario de imágenes con 8 características, y diccionario de imágenes originales. 
        El "key" es el número de fotograma.
    '''
    originals = {}
    images = {}
    for i, image_df in images_data:
        k = int(i)
        image_df.drop(['f'], axis = 1, inplace = True)

        # Almacena originales
        original_shape = original_data[k][:,:,:3].shape
        originals[k] = image_df[['R', 'G', 'B']].to_numpy().reshape(original_shape)

    
        if norm:
            stscaler = StandardScaler().fit(image_df)
            np_datos_escalados = stscaler.transform(image_df)
            images[k] = np_datos_escalados
            
    return images, originals


def preprocess(original):
    '''
    Preprocesa una imagen única: 
        - Aplica suavizado gaussiano de 5x5
        - Extrae la característica de color HSV y posición i,j de cada pixel
        - Crea el DataFrame de la librería "crea_tabla.py" con 8 columnas
            i,j,B,G,R,H,S,V


    Parameters:
    -----------
    original : array
        Array de los canales de color de la imagen con forma (altura, ancho, 3)

    Return:
    -------
    DataFrame    
        DataFrame donde cada fila es un pixel y cada columna una característica.
    '''

    ##### SUAVIZADO GAUSSIANO
    # Kernel Gaussiano 5x5
    kernel_gauss = np.array([[1,  4,  6,  4, 1],
                             [4, 16, 24, 16, 4],
                             [6, 24, 36, 24, 6],
                             [4, 16, 24, 16, 4],
                             [1,  4,  6,  4, 1]]) / 256


    # Aplicar la convolución 2D para cada canal
    image_smoothed = np.zeros_like(original)
    for i in range(3):
        image_smoothed[..., i] = convolve2d(original[..., i], kernel_gauss, boundary='symm', mode='same')

    image_smoothed = image_smoothed.astype(np.uint8)


    ##### TRANSFORMACIÓN
    # Expresar resultado en 6 canales (RGB|HSV)
    image_smoothed_bgr = image_smoothed[...,[2,1,0]]
    cuadro_hsv = cv.cvtColor(image_smoothed_bgr, cv.COLOR_BGR2HSV)
    image_smoothed_6channel = np.dstack((image_smoothed_bgr, cuadro_hsv))


    # Obtener expresión adecuada para el algoritmo
    image_4d = image_smoothed_6channel[np.newaxis, ...]
    arreglo_datos = crea_arreglo_datos(image_4d)

    # Tabla con características y derivados
    tabla_datos = pd.DataFrame(data=arreglo_datos,
                            columns=('f','i','j','B','G','R','H','S','V'))
    tabla_datos.drop(['f'], axis = 1, inplace = True)
    return tabla_datos

def varianza_acumulada(dataset, ax):
    '''
    Grafica la varianza acumulativa en cada número de componentes.

    Parameters:
    -----------
    dataset : array, DataFrame
        Array de los datos donde cada columna es una característica.

    '''
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    sc = StandardScaler()
    X_std = sc.fit_transform(dataset)

    # Paso 2: Obtener matriz de covarianza
    cov_mat = np.cov(X_std.T)

    # Paso 3: Descomponer la matriz de covarianza en eigenvectores y eigenvalores
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # print('Eigenvals : ', eigen_vals)
    # print('Eigenvects : ', eigen_vecs)

    # Paso 4: Ordenar eigenvalores de manera decreciente

    tot = sum(eigen_vals) # Suma de eigenvals
    var_exp = [ev/tot for ev in sorted(eigen_vals, reverse = True)]

    # Paso 5: Seleccionar los k eigenvectores con los k mayores eigenvalores; con k la dimensión de un nuevo subespacio

    # Gráfica
    cum_var_exp = np.cumsum(var_exp)
    ax.bar(range(1,X_std.shape[1] + 1 ), var_exp, label = 'varianza individual', align = 'center')
    ax.step(range(1,X_std.shape[1] + 1), cum_var_exp, where = 'mid', label = 'varianza acumulativa')
    ax.legend(loc = 'best')
    ax.set_ylabel('Varianza')
    # plt.show()