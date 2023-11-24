import onnxruntime as ort
import numpy as np
import cv2
import time
from Onnx_Class import CnnOnnx
from sort import Sort  # Assuming SORT is in a file named 'sort.py'

# Your CnnOnnx class and other helper functions go here

# Initialize your detector
detector = CnnOnnx('YoloV7_onnx/yolov7_dynamic.onnx', size=640, cuda=True)

# Initialize SORT tracker
tracker = Sort()  # You can adjust parameters like max_age, min_hits, etc.


# Function to process a video frame
def process_frame(frame):
    # Detect objects in the frame
    boxes, scores, classes = detector.detect_image(frame)

    # Format detections for SORT (x1, y1, x2, y2, score)
    detections = np.array([box + [score] for box, score in zip(boxes, scores)])

    # Update SORT tracker with current frame detections
    tracked_objects = tracker.update(detections)

    # Draw tracked objects on the frame
    for x1, y1, x2, y2, track_id in tracked_objects:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),
                    2)

    return frame


# Process video (example using a video file)
video_path = 'vid1.mp4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
output_video_path = 'output_video.mp4'  # Output file with .mp4 extension
writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Tracked Objects', processed_frame)
    writer.write(processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
