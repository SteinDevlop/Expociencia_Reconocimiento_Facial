import cv2 as cv
import numpy as np
import os
#import imutils
statuo_modelo = input("""
Seleccione status del modelo segun su numero:
1-Estudiante
2-Docente
3-Empleado
4-Administrador
5-Visitante

""")
identificador_modelo = input("Indique el identificador del modelo: ")

if statuo_modelo == "1":
    statuo_modelo = "EST"
if statuo_modelo == "2":
    statuo_modelo = "DOC"
if statuo_modelo == "3":
    statuo_modelo = "EMP"
if statuo_modelo == "4":
    statuo_modelo = "ADM"
if statuo_modelo == "5":
    statuo_modelo = "VSI"
else:

   pass


nombre_modelo = print(f"Modelo Registrado como: {statuo_modelo}{identificador_modelo}")
modelo = statuo_modelo + identificador_modelo
ruta1 = "Data_Training"
rutacompleta = ruta1 + "/" + modelo
if not os.path.exists(rutacompleta):
 os.makedirs(rutacompleta)


ruidos = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
camara = cv.VideoCapture(0)
id = 0
while True:
    respuesta,Captura = camara.read()
    if respuesta == False: break
    #Captura=imutils.resize(Captura,width=610)
    grises = cv.cvtColor(Captura,cv.COLOR_BGR2GRAY)
    idcaptura = Captura.copy()


    cara = ruidos.detectMultiScale(grises,1.2,7)
    for(x,y,e1,e2) in cara:
        cv.rectangle(Captura, (x, y), (x+e1,y+e2), (166,45,236),2)
        rostrocapturado = idcaptura[y:y+e2,x:x+e1]
        rostrocapturado = cv.resize(rostrocapturado, (160,160), interpolation= cv.INTER_CUBIC)
        cv.imwrite(rutacompleta + "/imagen_{}.jpg".format(id),rostrocapturado)
        id = id+1
    cv.imshow("Resultado Rostro",Captura)

    if id == 1000:
        break
    if cv.waitKey(1) == ord("x"):
        break
camara.release()
