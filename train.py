import cv2
import os
import numpy as np

# Path to the dataset
dataset_path = "./images"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prepare training data
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
        if img is None:
            continue  # Skip files that are not images or are unreadable

        img_np = np.array(img, "uint8")
        id = int(os.path.split(image_path)[-1].split("_")[0])  # Assuming filenames are like id_imageNumber.jpg
        faces = detector.detectMultiScale(img_np)

        if len(faces) == 0:
            print(f"No face detected in {image_path}")
            continue  # Skip images with no faces detected

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids

# Start training
print("Training faces...")
faces, ids = get_images_and_labels(dataset_path)

if len(faces) == 0:
    print("No faces found for training.")
else:
    recognizer.train(faces, np.array(ids))
    recognizer.write("trainer.yml")
    print("Training completed!")
