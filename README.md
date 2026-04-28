# hay-bales

Small project about finding hay bales using YOLO object detection. Ideally would like to eventually get a lightweight working version to run live on a small drone.

## Overview

This project uses YOLO (You Only Look Once) to detect hay bales in images and video.

## Project Structure

- `notebooks/` - Jupyter notebooks for exploration and training
- `src/` - Source code and utilities
- `weights/` - Trained model weights

## Getting Started

1. **Clone the repository and set up the environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download Model Weights**:
   The `*.pt` weights are too large for Git so they are excluded. Please download `best.pt` from your data source and place it in the `weights/` directory.

3. **Inference / Detection**:
   To analyze a new image, video, or entire directory, use the detection script.
   ```bash
   python src/detect.py --source prueba2.jpg --weights weights/best.pt
   ```
   **Optional Arguments:**
   - `--source`: Path to a single image, video file, or directory of images (default: `prueba.png`).
   - `--conf`: Sets the confidence threshold (e.g., `--conf 0.7`) to filter out weak detections (default: `0.5`).
   - `--show`: Include this flag to display the output in a popup window (omit for headless environments like drones or servers).

   **Example Output:**
   ```
   Cargando el cerebro artificial desde: weights/best.pt...
   Analizando la imagen: prueba2.jpg...

   image 1/1 C:\Users\tomas\Documents\Projects\hay-bales\prueba2.jpg: 384x640 5 hay - v6 2023-10-25 6-45pms, 102.6ms
   Speed: 4.1ms preprocess, 102.6ms inference, 2.6ms postprocess per image at shape (1, 3, 384, 640)
   Results saved to C:\Users\tomas\Documents\Projects\hay-bales\runs\detect\predict-5
   ¡Análisis completado! La imagen con las cajas está en: C:\Users\tomas\Documents\Projects\hay-bales\runs\detect\predict-5
   ```
   The script loads the trained model, detects the hay bales (finding 5 in this example), and saves a new image with bounding boxes drawn over the detections in the automatically generated `runs/detect/predict-*/` directory.

4. **Training / Exploration**:
   Run the environment with Jupyter to open the notebooks in `notebooks/`:
   ```bash
   jupyter notebook
   ```
