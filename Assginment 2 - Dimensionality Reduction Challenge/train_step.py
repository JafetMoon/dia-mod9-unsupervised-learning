import numpy as np
import time as time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def dbscan(datos, original, ax_row):
    '''
    datos : Array de características de los datos (n,8)
    original: Array de la imagen original  con shape (h,w,3)
    '''
    ## Estandarización
    stscaler = StandardScaler().fit(datos)
    np_datos_escalados = stscaler.transform(datos)

    # Entrenamiento de DBSCAN
    min_samples = int(len(np_datos_escalados) * 0.01)
    t_ini = time.time()
    dbsc = DBSCAN(eps = 0.5, min_samples = min_samples).fit(np_datos_escalados)
    print(f"Tiempo transcurrido: {time.time() - t_ini:.3f}s")


    # Máscaras resultantes
    labels = dbsc.labels_
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True

    # Visualización
    rens = original.shape[0]
    cols = original.shape[1]
    grupos = labels.reshape((rens, cols))

    ax_row[0].imshow(original.astype(int))
    ax_row[1].imshow(grupos)
    return labels

def kmeans(datos, original, ax_row):
    '''
    datos : Array de características de los datos (n,8)
    original: Array de la imagen original  con shape (h,w,3)
    '''
    ## Estandarización
    stscaler = StandardScaler().fit(datos)
    np_datos_escalados = stscaler.transform(datos)

    # Entrenamiento de DBSCAN
    min_samples = int(len(np_datos_escalados) * 0.01)
    t_ini = time.time()
    dbsc = DBSCAN(eps = 0.5, min_samples = min_samples).fit(np_datos_escalados)
    print(f"Tiempo transcurrido: {time.time() - t_ini:.3f}s")


    # Máscaras resultantes
    labels = dbsc.labels_
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True

    # Visualización
    rens = original.shape[0]
    cols = original.shape[1]
    grupos = labels.reshape((rens, cols))

    ax_row[0].imshow(original.astype(int))
    ax_row[1].imshow(grupos)
    return labels