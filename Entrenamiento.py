import cv2 as cv
import numpy as np
import os
from time import time
Data_ruta = "Data_Training"
listData = os.listdir(Data_ruta)
ids=[]
rostrosData=[]
id = 0
timpoinicial = time()
accion = input("""INICIANDO ENTRENAMIENTO
El proceso puede durar desde minutos hasta horas
¿desea proceder?
SELECCIONE Y PARA INICIAR
SELECCIONE CUALQUIER OTRA LETRA PARA FINALIZAR
""")
if accion == "Y":
    for fila in listData:
        rutacompleta = Data_ruta +"/"+ fila
        print("Iniciando lectura ...")
        for archivo in os.listdir(rutacompleta): 
        
            print("imagen: ", fila + "/" + archivo)
            ids.append(id)
            rostrosData.append(cv.imread(rutacompleta + "/" + archivo, 0))
        id = id + 1
        tiempofinallectura = time()
        tiempolecturatotal = tiempofinallectura - timpoinicial
        print("Tiempo total: ", tiempolecturatotal)
    entrenamientoModelo1 = cv.face.EigenFaceRecognizer_create()
    print("Iniciando el entrenamiento ...")
    entrenamientoModelo1.train(rostrosData, np.array(ids))
    tiempofinalentrenamiento = time()
    tiempototalentrenamiento = tiempofinalentrenamiento-tiempolecturatotal
    print("Tiempo entrenamiento total ", tiempototalentrenamiento)
    entrenamientoModelo1.write("EntrenamientoEigenFaceRecognizer.xml")
    print("entrenamiento concluido")

else:
    exit()
