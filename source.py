
# ---------------------------------- importing Libraries ---------------------------------- #

#import os
import csv
import easyocr
import cv2                      
import torch
#import time

# ---------------------------------- Load YOLO Model ---------------------------------- #

MODEL_PATH = "/home/ayaan/Desktop/numberplate/model.pt"
# if the model fails to load (e.g., file missing or corrupted or wrong path)we stop the program and show a clear message
try:
    model = torch.hub.load("ultralytics/yolov3","custom",path=MODEL_PATH,force_reload=True)
    print("[INFO] YOLO model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    exit()

# ---------------------------------- Initialize EasyOCR ---------------------------------- #

# If OCR setup fails (maybe due to missing files or incompatible setup),script stop 
try:
    reader = easyocr.Reader(["en"])
    print("[INFO] EasyOCR reader initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize EasyOCR reader: {e}")
    exit()

# ---------------------------------- Video Input & Output ---------------------------------- #

video =cv2.VideoCapture("path_of_input_vidoe.mp4")#write path of input video
frame_width=int(video.get(3))
frame_height=int(video.get(4))
fps=int(video.get(cv2.CAP_PROP_FPS))
output_video=cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"),fps,(frame_width,frame_height))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CSV file Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

csv_path = "ocr_results.csv" #CSV path here
# Make sure the CSV file is created properly,if there is a problem(like permission issues)
try:
    with open(csv_path, mode='w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "X1", "Y1", "X2", "Y2", "Confidence", "Detected Text"])
except Exception as e:
    print(f"[ERROR] Failed to create CSV file: {e}")
    exit()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Frame Processing Loop~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

frame_count = 0
while video.isOpened():
    # Wrapping each frame in a try block to catch anything unexpected that may break the loop
    try:
        ret, frame = video.read()
        if not ret:
            print("[INFO] End of video or frame read failed.")
            break
        frame_count+=1


        # If YOLO model fails to detect objects in a frame, we skip that frame and move on without stopping the script.
        try:
            results = model(frame)
            detections = results.pandas().xyxy[0]
        except Exception as e:
            print(f"[ERROR] YOLO detection failed at frame {frame_count}: {e}")
            continue


        for _, row in detections.iterrows():
            # We’re checking each detected object carefully, in case something goes wrong while extracting box info.
            try:
                x1,y1,x2,y2 = int(row["xmin"]),int(row["ymin"]),int(row["xmax"]) ,int(row["ymax"])
                conf = float(row["confidence"])
                cls = int(row["class"])

                if cls!=0:# Only process class 0 (assumed to be license plates)
                    continue
                plate=frame[y1:y2,x1:x2]


                if plate.size == 0 or plate.shape[0]<10 or plate.shape[1]< 10:
                    print(f"[WARNING] Skipped small or invalid crop at frame {frame_count}")
                    continue
                """preparing plate for OCR"""
                gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY),_
                thresh = cv2.threshold(gray ,150,255,cv2.THRESH_BINARY)
                try:
                    text = reader.readtext(thresh, detail=0)
                except Exception as ocr_error:
                    print(f"[ERROR] OCR failed at frame {frame_count}, box [{x1}, {y1}, {x2}, {y2}]: {ocr_error}")
                    text = ["<OCR Error>"]
                text_str = " ".join(text)
                print(f"[INFO] Frame {frame_count} | Text: {text_str} | Confidence: {conf:.2f}")


                """write OCR result into .csv"""
                with open(csv_path,mode='a',newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([frame_count ,x1,y1,x2, y2,f"{conf:.2f}", text_str])

                # Draws box and detected text
                cv2.rectangle(frame,(x1, y1),(x2, y2),(0,180,255)) #reddish yellow box  or saffaron box
                cv2.putText(frame, text_str,(x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 180, 255))

            except Exception as box_error:
                print(f"[ERROR] Bounding box processing error at frame {frame_count}: {box_error}")


                """Display the frame"""
        output_video.write(frame)
        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            print("[INFO] Processing stopped by user.")
            break
    except Exception as frame_error:
        print(f"[ERROR] Unexpected error at frame {frame_count}: {frame_error}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cleaning up ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

video.release()
output_video.release()
cv2.destroyAllWindows()
print("[INFO] Video processing completed. OCR results saved to ocr_results.csv")
