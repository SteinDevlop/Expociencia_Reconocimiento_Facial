import tkinter as tk
from tkinter import Label, PhotoImage, ttk
import subprocess
from tkinter.constants import INSERT
import cv2 as cv
import numpy as np
import os
from time import time
import shutil
from os import remove
from PIL import Image
import webbrowser
ventana = tk.Tk()
#tamaño
ventana.geometry("1200x800")
ventana.resizable(width=False, height=False)
ventana.title("Expociencia 2022")

ruta_madre_P1 = os.getcwd()
ruta_iconbit_P1 = ruta_madre_P1 + r"/Data_Image/HK_038.xbm"
ventana.iconbitmap("@"+ruta_iconbit_P1)
ruta_fondo_P1 = PhotoImage(file="Data_Image/window_contraste.png")
ventana.tk.call('wm', 'iconphoto', ventana._w, ruta_fondo_P1)
background = Label(image=ruta_fondo_P1, text="")
background.place(x=0, y=0, relwidth=1, relheight=1)
#definimos el evento (command)
def iniciar_generador():
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
    nombre_modelo = print(
    f"Modelo Registrado como: {statuo_modelo}{identificador_modelo}")
    modelo = statuo_modelo + identificador_modelo
    ruta1 = "Data_Training"
    rutacompleta = ruta1 + "/" + modelo
    if not os.path.exists(rutacompleta):
     os.makedirs(rutacompleta)
    ruidos = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    camara = cv.VideoCapture(0)
    id = 0
    while True:
        respuesta, Captura = camara.read()
        if respuesta == False:
            break
        grises = cv.cvtColor(Captura, cv.COLOR_BGR2GRAY)
        idcaptura = Captura.copy()

        cara = ruidos.detectMultiScale(grises, 1.2, 7)
        for(x, y, e1, e2) in cara:
            cv.rectangle(Captura, (x, y), (x+e1, y+e2), (166, 45, 236), 2)
            rostrocapturado = idcaptura[y:y+e2, x:x+e1]
            rostrocapturado = cv.resize(
                rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)
            cv.imwrite(rutacompleta + "/imagen_{}.jpg".format(id), rostrocapturado)
            id = id+1
        cv.imshow("Resultado Rostro", Captura)

        if id == 501:
            break
        if cv.waitKey(1) == ord("x"):
            break
    camara.release()
    cv.destroyAllWindows()
def promotor():
    def cerrar():
        ventana.destroy()
    ruta_madre_promo = os.getcwd()
    ruta_promo_img = ruta_madre_promo + r"/Data_Image/CC-2224.png"
    webbrowser.open(ruta_promo_img)
def iniciar_entrenamiento():
 Data_ruta = "Data_Training"
 listData = os.listdir(Data_ruta)
 ids=[]
 rostrosData=[]
 id = 0
 timpoinicial = time()
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
 print("Entrenamiento concluido, Puede continuar")
def resultado():
 Data_ruta = "Data_Training"
 listData = os.listdir(Data_ruta)
 entrenamientoModelo1 = cv.face.EigenFaceRecognizer_create()
 entrenamientoModelo1.read("EntrenamientoEigenFaceRecognizer.xml")
 ruidos=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
 camara=cv.VideoCapture(0)
 while True:
     _,captura = camara.read()
     grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
     idcaptura = grises.copy()
     cara = ruidos.detectMultiScale(grises,1.2,7) 
     for(x, y, e1, e2) in cara:
         rostrocapturado = idcaptura[y:y+e2, x:x+e1]
         rostrocapturado = cv.resize(rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)
         resultado = entrenamientoModelo1.predict(rostrocapturado)
         cv.putText(captura, "{}".format(resultado), (x, y-20),2, 1.1, (12, 232, 217), 1, cv.LINE_AA)
         if resultado[1]< 9000:
             nombre = cv.putText(captura, "{}".format(listData[resultado[0]]), (x, y-5),1, 1.3, (12, 232, 217), 2, cv.LINE_AA)
             cv.rectangle(captura, (x, y), (x+e1, y+e2), (206, 0, 213), 2)
         else:
             cv.putText(captura, "Desconocido", (x, y-5), 1,
                        1.3, (12, 12, 212), 2, cv.LINE_AA)
             cv.rectangle(captura, (x, y), (x+e1, y+e2), (255, 0, 0), 2)
     cv.imshow("resultado",captura)
     if cv.waitKey(1) == ord("x" or "x"):
         break
 camara.release()
 cv.destroyAllWindows()
def opcion():
    #configuraciones de ventanas
    ventana_opcion = tk.Tk()
    ventana_opcion.geometry("250x200")
    ventana_opcion.resizable(width=False, height=False)
    ventana_opcion.title("Configuracion")
    ruta_madre_opcion = os.getcwd()
    ruta_iconbit_opcion = ruta_madre_opcion + r"/Data_Image/config_icon.xbm"
    ventana_opcion.iconbitmap("@"+ruta_iconbit_opcion)
    #Funciones de los botones 
    def borrar_datos():
        datatraining = os.path.isdir("Data_Training")
        folder = "Data_Training"
        entrenamientoxml = os.path.isfile(
            "EntrenamientoEigenFaceRecognizer.xml")
        if entrenamientoxml == True:
            remove("EntrenamientoEigenFaceRecognizer.xml")
        else:
            pass
        if datatraining == True:
           for file in os.listdir(folder):
                file_path = os.path.join(folder,file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    shutil.rmtree(file_path)
        if datatraining == True and entrenamientoxml == True:
            reset_boton.config(text='Base de datos reseteada y archivo .xml borrado')
        elif datatraining == True and entrenamientoxml == False:
            reset_boton.config(text='Base de datos reseteada')
        elif datatraining == False and entrenamientoxml == True:
            reset_boton.config(text='Base de datos no encontrada, Archivo .xml borrado')
        elif datatraining == False and entrenamientoxml == False:
            reset_boton.config(
                text='Base de datos no encontrada, Archivo .xml no encontrado')
        else:
            reset_boton.config(
                text='Error desconocido')


    def error_type():
        ventana_guide = tk.Tk()
        ventana_guide.geometry("900x500")
        ventana_guide.title("Resolver error")
        ruta_madre_guide = os.getcwd()
        ruta_iconbit_guide = ruta_madre_guide + r"/Data_Image/config_icon.xbm"
        ventana_guide.iconbitmap("@"+ruta_iconbit_guide)
        mensaje = tk.Text(ventana_guide, background="white", width=165, height=25)
        mensaje.grid(row=1, column=2, sticky="ENS")
        mensaje.insert(INSERT,"""
        ------------------GENERADOR------------------
Si ocurre un error inesperado que impida prender la cámara en 
este punto, verifique que no haya alterado ninguna carpeta o
archivo que vino al conjunto con este EXE.

Generalmente se arregla inicializando "Borrar base de datos" 
(Opciones > Borrar base de datos). Si la opción no sirve, 
desde la carpeta "Data_Training" borre todo el contenido 
de la carpeta. Si ninguna de las dos soluciones ayudan, 
reinstale los componentes o en su defecto, todo el programa 
en su totalidad.

------------------ENTRENAMIENTO------------------
Si ocurre un error inesperado que impida que el ultimo mensaje 
"Entrenamiento concluido, Puede continuar" aparezca:

- Verifique que no tenga imágenes en la carpeta 
"Data_Training" sin estar en su respectiva carpeta

- Si tomo muchas fotos o paso por el Generador 
múltiples veces, el proceso puede ser mas largo 
de lo previsto (puede esperar a que el proceso termine o cerrarlo)

- Si es el caso de que tenga fotos en la 
carpeta "Data_Training" directamente, puede borrarlas 
entrando a la carpeta o en 
la configuración de base de datos.

------------------RECONOCIMIENTO------------------
Si ocurre un error inesperado que impida que se prenda la cámara:

- Verifique que no tenga imágenes en la carpeta "Data_Training" 
sin estar en su respectiva carpeta.
- Verifique que haya pasado por los dos procesos anteriores 
secuencialmente.""")
    def Review():
        webbrowser.open("https://forms.gle/LM6ZzmsTwnmSBbcy6")
    #botones
    reset_boton = ttk.Button(ventana_opcion, text="Borrar base de datos", command = borrar_datos)
    info_error = ttk.Button(ventana_opcion, text="Resolver error", command = error_type)
    reseña = ttk.Button(ventana_opcion, text = "Reseña", command = Review)
    #Ubicacion de los botones
    reset_boton.place(relx=0.25, rely=0.1)
    info_error.place(relx = 0.25, rely = 0.2)
    reseña.place(relx=0.25, rely=0.3)
    reset_boton.grid(row=1, column=1, sticky="ENSW")
    info_error.grid(row=5, column=1, sticky="ENW")
    reseña.grid(row=7, column=1, sticky="ENSW")
def guide():
    ventana_guide = tk.Tk()
    ventana_guide.geometry("800x400")
    ventana_guide.resizable(width=False, height=False)
    ventana_guide.title("Guia")
    ruta_madre_guide = os.getcwd()
    ruta_iconbit_guide = ruta_madre_guide + r"/Data_Image/config_icon.xbm"
    ventana_guide.iconbitmap("@"+ruta_iconbit_guide)
    mensaje = tk.Text(ventana_guide, background="white", width=165, height=25)
    mensaje.grid(row=1, column=2, sticky="ENS")
    mensaje.insert(INSERT, """
Los botones deben ejecutarse en orden:

   1 .Generador
   2. Entrenamiento
   3. Reconocimiento

1. Al ejecutar "Generador": Ingresa los datos que se solicita para indicar el rango del modelo. 
Le damos un nombre o código identificatorio y procederá a activar la cámara.
Tomará 500 fotos, luego se cerrará o puedes adelantarlo pulsando tecla x (en minúscula).

2. Al ejecutar "Entrenamiento": Leerá y entrenará el sistema, debes ser paciente 
hasta que el proceso finalicé.

3. Al ejecutar el "Reconocimiento": Mostrará una cámara que identificará al modelo. Puedes
cerrarla con "x".
""")

generador_boton = ttk.Button(ventana, text = "Generador", command=iniciar_generador)
entrenamiento_boton = ttk.Button(ventana, text = "Entrenamiento", command=iniciar_entrenamiento)
resultado_boton = ttk.Button(ventana, text = "Reconocimiento", command=resultado)
promocion_boton = ttk.Button(ventana, text="¡Obtenlo!", command=promotor)
opciones_boton = ttk.Button(ventana, text="Configuracion", command=opcion)
guia_boton = ttk.Button(ventana, text="¿Como lo uso?", command=guide)
entrenamiento_boton.grid(row=2, column=2, sticky="ENS")
resultado_boton.grid(row=3, column=2, sticky="ENS")
generador_boton.grid(row=1, column=2, sticky="ENS")
opciones_boton.grid(row=4, column=2, sticky="ENS")
guia_boton.grid(row=1, column=3, sticky="ENS")
generador_boton.place(
    relx=0.28, rely=0.20, width=200, height=100)
entrenamiento_boton.place(
    relx=0.28, rely=0.35, width=200, height=100)
resultado_boton.place(
    relx=0.28, rely=0.50, width=200, height=100)
guia_boton.place(
    relx=0.28, rely=0.65, width=200, height=100)
opciones_boton.place(
    relx=0.60, rely=0.84, width=200, height=50)
promocion_boton.place(
    relx=0.80, rely=0.84, width=200, height=50)
ventana.mainloop()

