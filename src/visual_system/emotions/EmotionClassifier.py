import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class EmotionClassifier:
    def __init__(self, model_path: str):
        self.model = torch.load(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])
        self.emotion_dict = {
            0: "Angry",
            1: "Disgusted",
            2: "Fearful",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprised"
        }

    def load_and_preprocess_image(self, image):
        """Load and preprocess an image."""
        # Check if image is a file path
        if isinstance(image, str) and os.path.isfile(image):
            image = Image.open(image)

        # Check if image is a numpy array
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convert to grayscale and apply transformations
        image = image.convert('L')  # Convert to grayscale
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension
        return image

    def _predict_single_image(self, image):
        """Predict the emotion of a single image."""
        image = self.load_and_preprocess_image(image)
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def predict_emotions(self, images: list):
        """Predict emotions for a list of images."""
        emotions_out = []
        for image in images:
            prediction = self._predict_single_image(image)
            emotions_out.append(self.emotion_dict[prediction.item()])
        return emotions_out


# Usage example
if __name__ == '__main__':
    classifier = EmotionClassifier('best.pt')
    images = ['im1.png', 'im2.png']  # These can be paths, PIL images, or numpy arrays
    res = classifier.predict_emotions(images)
    print(res)
