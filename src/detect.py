import argparse
from ultralytics import YOLO

def main(weights_path, image_path):
    # 1. Cargamos el modelo local
    print(f"Cargando el cerebro artificial desde: {weights_path}...")
    modelo = YOLO(weights_path)

    # 2. Le pasamos la imagen de prueba
    print(f"Analizando la imagen: {image_path}...")
    # El parámetro save=True le dice que dibuje las cajas y guarde la foto
    resultados = modelo(image_path, save=True)

    # 3. Mostrar la ruta donde se guardó
    ruta_salida = resultados[0].save_dir
    print(f"¡Análisis completado! La imagen con las cajas está en: {ruta_salida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para detectar objetos (fardos) usando YOLOv8.")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Ruta a los pesos del modelo (*.pt)")
    parser.add_argument("--image", type=str, default="prueba.png", help="Ruta a la imagen para analizar")
    
    args = parser.parse_args()
    main(args.weights, args.image)
