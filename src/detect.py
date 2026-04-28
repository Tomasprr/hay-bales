import argparse
from ultralytics import YOLO

def main(weights_path, source_path, conf_thresh, show_img, save_img):
    print(f"Loading model from: {weights_path}...")
    model = YOLO(weights_path)

    print(f"Analyzing: {source_path}...")
    results = model(
        source=source_path, 
        save=save_img, 
        show=show_img,
        conf=conf_thresh,
        iou=0.3, # Avoid overlapping detections
        line_width=1,
        show_labels=False
    )

    if results:
        if save_img:
            output_path = results[0].save_dir
            print(f"Analysis complete! Results saved to: {output_path}")
        else:
            total_objects = sum(len(r.boxes) for r in results if r.boxes)
            print(f"Analysis complete! Processed {len(results)} image(s) and found {total_objects} total objects. (Saving is disabled)")
    else:
        print("No results found or the source is empty.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to detect hay bales using YOLOv8.")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to the model weights (*.pt)")
    parser.add_argument("--source", type=str, default="prueba.png", help="Path to the image, video, or directory to analyze")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection (0.0 to 1.0)")
    parser.add_argument("--show", action="store_true", help="Show the image/video during analysis")
    parser.add_argument("--save", action="store_true", help="Save the image/video with drawn bounding boxes")
    
    args = parser.parse_args()
    main(args.weights, args.source, args.conf, args.show, args.save)
