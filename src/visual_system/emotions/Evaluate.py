import os
import cv2
import numpy as np
from PIL import Image
from EmotionClassifier import EmotionClassifier  # Import the EmotionClassifier class


def select_roi(image_path):
    """
    Allows the user to select regions of interest on an image.
    Returns a list of selected ROIs along with their coordinates.
    """
    image = cv2.imread(image_path)
    # Instructions
    print("Select ROIs. Press 'Enter' or 'Space' when done. Press 'c' to cancel the last selection.")
    rois = []
    while True:
        roi = cv2.selectROI("Select Rois", image)

        if roi == (0, 0, 0, 0):
            break
        else:
            rois.append(roi)
    cv2.destroyAllWindows()
    roi_images = [(image[y:y + h, x:x + w], (x, y, w, h)) for (x, y, w, h) in rois]
    return roi_images


def draw_emotions_on_image(image, roi_info, emotions):
    """
    Draws the ROIs and corresponding emotions on the image.
    """
    for (roi, coords), emotion in zip(roi_info, emotions):
        x, y, w, h = coords
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def main(image_paths):
    classifier = EmotionClassifier('best.pt')
    for image_path in image_paths:
        roi_info = select_roi(image_path)
        roi_images = [Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) for roi, _ in roi_info]
        emotions = classifier.predict_emotions(roi_images)
        print(f"Emotions in {image_path}: {emotions}")

        # Draw emotions on the image and display it
        original_image = cv2.imread(image_path)
        draw_emotions_on_image(original_image, roi_info, emotions)
        cv2.imshow("Emotion Results", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    folder = 'images'
    image_paths = [os.path.join(folder, image_path) for image_path in os.listdir(folder)]
    main(image_paths)
