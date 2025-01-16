import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pdb
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from box import Box
from loguru import logger


def crop_face(image_path, mtcnn, device):
    """Crops the largest detected face from the image using OpenCV."""
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Detect faces and select the largest one
    boxes, _ = mtcnn.detect(img_rgb)
    if boxes is None:
        print(f"No face detected in image: {image_path}")
        return None

    # Select the largest face
    largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = map(int, largest_box)
    cropped_img = img_rgb[y1:y2, x1:x2]
    return cropped_img

def load_model(device, checkpoint_path):
    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    logger.info("Pre-trained model loaded successfully without classifier layer.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Checkpoint loaded successfully.")
    return model

def get_face_embedding(cropped_img, transform, model, device):
    if cropped_img is None:
        return None
    img_resized = cv2.resize(cropped_img, (160, 160))
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        face_embedding = model(img_tensor).cpu()
    return face_embedding

def compare_faces(embedding1, embedding2, threshold=0.9422):
    distance = (embedding1 - embedding2).norm().item()
    logger.info(f"distance: {distance}")
    prediction = "match" if distance < threshold else "mismatch"
    return distance, prediction

def visualize_comparison_org(image_path1, image_path2, prediction):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1_rgb)
    axes[0].set_title(f"Image 1: {prediction}", fontsize=16)
    axes[0].axis('off')
    axes[1].imshow(img2_rgb)
    axes[1].set_title(f"Image 2: {prediction}", fontsize=16)
    axes[1].axis('off')
    plt.show()

def visualize_comparison(image_path1, image_path2, prediction):
    """Visualizes the comparison by overlaying the result on the images."""
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Create a 1x2 grid of subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    axes[0].imshow(img1)
    axes[0].set_title(f"Image 1: {prediction}", fontsize=20)
    axes[0].axis('off')  # Hide the axis

    axes[1].imshow(img2)
    axes[1].set_title(f"Image 2: {prediction}", fontsize=20)
    axes[1].axis('off')  # Hide the axis

    plt.tight_layout()  # Adjust spacing to make it compact
    # plt.tight_layout()

    # Save the figure as a .jpg file
    plt.savefig('Demo_image3_tuple.jpg', format='jpg', dpi=300)
    plt.show()

def evaluate_two_images(image_path1, image_path2, mtcnn, model, transform, device, threshold=0.9422):
    if mtcnn is not None:
        cropped_img1 = crop_face(image_path1, mtcnn, device)
        cropped_img2 = crop_face(image_path2, mtcnn, device)
        if cropped_img1 is None or cropped_img2 is None:
            logger.error("One or both images could not be processed.")
            return
    # pdb.set_trace()
    embedding1 = get_face_embedding(cropped_img1, transform, model, device)
    embedding2 = get_face_embedding(cropped_img2, transform, model, device)
    if embedding1 is None or embedding2 is None:
        logger.error("One or both embeddings could not be generated.")
        return
    distance, prediction = compare_faces(embedding1, embedding2, threshold)
    print(f"Distance: {distance:.4f}, Prediction: {prediction}")
    visualize_comparison(image_path1, image_path2, prediction)

if __name__ == "__main__":
    with open("config.yaml", "r") as config_file:
        config_dict = yaml.safe_load(config_file)
        config = Box(config_dict)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Available Device: {device}")

    if config.inference.fd_model_run:
        fd_mtcnn = MTCNN(device=device)  # Initialize MTCNN for face detection
    else:
        fd_mtcnn = None
    # pdb.set_trace()

    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert from numpy to PIL for torchvision compatibility
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.train.transform.normalize.mean,
                             std=config.train.transform.normalize.std)
    ])

    checkpoint_path = os.path.join(config.inference.weight_dir,
                                   config.inference.checkpoint_name)
    model = load_model(device, checkpoint_path)

    # image_path1 = "docs/images/test/ID5239_000043.jpg"
    image_path1 = "docs/images/test/ID5230_000042.jpg"
    image_path2 = "docs/images/test/ID5230_002825.jpg"

    threshold = config.inference.opt_threshold
    evaluate_two_images(image_path1,
                        image_path2,
                        fd_mtcnn,
                        model,
                        transform,
                        device,
                        threshold=threshold)
