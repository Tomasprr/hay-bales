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
   To analyze a new image, use the detection script.
   ```bash
   python src/detect.py --image prueba.png --weights weights/best.pt
   ```

4. **Training / Exploration**:
   Run the environment with Jupyter to open the notebooks in `notebooks/`:
   ```bash
   jupyter notebook
   ```
