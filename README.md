# AI based License Plate Detection with OCR using YOLOv5

## Overview
This Python script is designed to detect license plates from a video using a pre-trained YOLOv5 model and then extract the plate numbers using EasyOCR. The code also saves the results in a CSV file and creates an annotated output video.

## Library Used
- OpenCV -> For handling video input/output and drawing annotations on frames.
- PyTorch -> For loading and running the YOLOv5 model.
- EasyOCR -> For performing Optical Character Recognition on detected plates.
- CSV -> For recording detection results.

## How it Works
1. The YOLOv5 model is loaded from the specified path. If the model fails to load,script stops with a proper error message.
2. EasyOCR is initialized for reading English text from detected license plates.
3. The video is opened and frame details like width, height, and FPS are captured.
4. A CSV file is prepared for logging OCR results.
5. Each frame is processed in a loop:
   -> The frame is passed to the YOLO model to detect license plates.
   -> Only objects classified as license plates are processed further.
   -> The detected license plate area is cropped and preprocessed.
   -> EasyOCR reads the text from the plate.
   -> The result is saved to the CSV file and annotated on the frame.
   -> The annotated frame is written to an output video file.
6. If any error occurs during detection or OCR, it’s caught and logged without stopping the script.
7. Once all frames are processed, the video and CSV file are saved, and resources are released.

## Variable Descriptions
- `MODEL_PATH` -> Path to the YOLOv5 model (.pt file).
- `model` -> Loaded YOLOv5 model used for detection.
- `reader` -> EasyOCR reader for recognizing text.
- `video` -> VideoCapture object for reading the video.
- `frame_width`,`frame_height` -> Dimensions of each video frame.
- `fps` -> Frames per second of the input video.
- `output_video` -> VideoWriter object to save the final video.
- `csv_path` -> Path for saving OCR results in CSV format.
- `frame_count` -> Tracks which frame is being processed.
- `ret, frame` -> Used to read frames from the video.
- `results` -> YOLO detection result per frame.
- `detections` -> DataFrame containing detection info.
- `x1,y1,x2,y2` -> Coordinates of the bounding box.
- `conf` -> Confidence score of the detection.
- `cls` -> Class ID from YOLO detection.
- `plate` -> Cropped image of the detected license plate.
- `gray` -> Grayscale version of the plate image.
- `thresh` -> Thresholded image used for OCR.
- `text` -> Raw text output from OCR.
- `text_str` -> Final formatted text string.

## Output Files
- `output_video.mp4` -> Video with bounding boxes and OCR results shown on each frame.
- `ocr_results.csv` -> File containing detection and OCR data for each frame.
