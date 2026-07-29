import cv2
import glob
import os

def cargar_imagenes_de_carpeta(carpeta):
    extensiones = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    rutas = []
    for ext in extensiones:
        rutas.extend(glob.glob(os.path.join(carpeta, ext)))
    rutas.sort()  # Opcional: ordena alfabéticamente
    imagenes = []
    for ruta in rutas:
        img = cv2.imread(ruta)
        if img is not None:
            imagenes.append(img)
        else:
            print(f"No se pudo cargar la imagen: {ruta}")
    return imagenes

def crear_panoramica(imagenes):
    stitcher = cv2.Stitcher_create()
    estado, panoramica = stitcher.stitch(imagenes)
    if estado == cv2.Stitcher_OK:
        print("¡Panorámica creada exitosamente!")
        cv2.imwrite("panoramica.jpg", panoramica)
        cv2.imshow("Panorámica", panoramica)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error al crear la panorámica. Código de error:", estado)

if __name__ == "__main__":
    carpeta = os.path.join(os.path.dirname(__file__), "fotos-base")
    print(f"Buscando imágenes en: {carpeta}")
    imagenes = cargar_imagenes_de_carpeta(carpeta)
    iamgenes = imagenes[1:4] # Se me queda trabado, esto es porque las fotos no son compatibles o la ram es muy mala
    if len(imagenes) < 2:
        print("Debes tener al menos dos imágenes en la carpeta 'fotos-base'.")
    else:
        crear_panoramica(imagenes)
