from ultralytics import YOLO
import torch
import gc

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    model = YOLO(r"C:\Users\Administrator\miniconda3\envs\sam\yolov8n.pt")

    model.train(
        data=r"H:/19/model/1.yaml",
        imgsz=(640, 3360),
        batch=8,
        epochs=100,
        device=0,
        rect=True,
        mosaic=False,
        workers=4,
        amp=True,
        cache="disk",
        mask_ratio=4,
        overlap_mask=True,
        lr0=0.0005,
        lrf=0.01,
        #sem=0.0,
        patience=15
    )