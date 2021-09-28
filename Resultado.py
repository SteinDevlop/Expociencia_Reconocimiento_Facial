import cv2 as cv
import os
Data_ruta = "Data_Training"
listData = os.listdir(Data_ruta)
entrenamientoModelo1 = cv.face.EigenFaceRecognizer_create()
entrenamientoModelo1.read("EntrenamientoEigenFaceRecognizer.xml")
ruidos=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
camara = cv.VideoCapture(0)
while True:
    _,captura = camara.read()
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura = grises.copy()
    cara = ruidos.detectMultiScale(grises,1.2,7)
    for(x, y, e1, e2) in cara:
        rostrocapturado = idcaptura[y:y+e2, x:x+e1]
        rostrocapturado = cv.resize(rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        resultado = entrenamientoModelo1.predict(rostrocapturado)
        cv.putText(captura, "{}".format(resultado),(x, y-20), 2, 1.1, (0, 255, 0),1,cv.LINE_AA)
        if resultado[1]< 9000:
            cv.putText(captura, "{}".format(listData[resultado[0]]), (x, y-5),1, 1.3, (0, 255, 255), 2, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2),(0, 255, 0),2)
        else:
            cv.putText(captura, "Desconocido", (x, y-5), 1, 1.3, (0, 0, 255), 2, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x+e1, y+e2), (0, 0, 255), 2)
    cv.imshow("resultado",captura)
    if cv.waitKey(1)== ord("x"):
        break
camara.release()
cv.destroyAllWindows()
