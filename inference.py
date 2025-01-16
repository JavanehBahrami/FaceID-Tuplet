import os
import torch
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import yaml

from torchvision import transforms
from box import Box
from loguru import logger
from facenet_pytorch import InceptionResnetV1
from PIL import Image


def load_model(device, checkpoint_path):

    # Load the pre-trained model without classification layer
    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    logger.info("pre-trained model is loaded successfully without classifier layer.")

    logger.info(f"Checkpoint path: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    logger.info("Checkpoint is loaded successfully.")

    return model


def get_face_embedding(image_path, transform, model, device):
    """Extracts the embedding of a given image using the specified transform."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Apply the same transformations as in train.py
    img_tensor = transform(img).unsqueeze(0).to(device)  # Shape: (1, 3, 160, 160)

    with torch.no_grad():
        face_embedding = model(img_tensor).cpu()

    return face_embedding


# Function to evaluate pairs and log results to a CSV
def compute_inference_metrics(predictions, labels):
    """
    Computes FPR, FNR, and raw counts (FP, FN) for predictions.

    Args:
        predictions: List of predicted labels (1 for match, 0 for mismatch).
        labels: List of ground-truth labels (1 for match, 0 for mismatch).

    Returns:
        metrics: Dictionary with FP, FN, FPR, FNR, TN, TP counts and rates.
    """
    # Initialize counts
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)  # True Positives
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)  # True Negatives
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)  # False Positives
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)  # False Negatives

    # Compute FPR and FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "FP": fp,
        "FN": fn,
        "FPR": fpr,
        "FNR": fnr,
        "TP": tp,
        "TN": tn
    }



def evaluate_and_log_pairs(embeddings, labels, folder_names, pairs_list, threshold=0.9422, output_csv="fv_results.csv"):
    """
    Evaluates all pairs using the given threshold and logs the results into a properly formatted CSV file.

    Args:
        embeddings: List of (embedding1, embedding2) pairs.
        labels: List of ground-truth labels (1 for match, 0 for non-match).
        folder_names: List of folder names corresponding to the embeddings.
        pairs_list: List of pair file paths corresponding to embeddings.
        threshold: Threshold to classify pairs as matches or mismatches.
        output_csv: Path to save the output CSV file.

    Returns:
        None
    """
    print("\n--- Evaluating Pairs with Threshold: {:.2f} ---".format(threshold))
    
    rows = []
    print(f"--- labels: {labels}")
    for idx, ((embedding1, embedding2), label, folder_name, str_pair) in enumerate(zip(embeddings, labels, folder_names, pairs_list)):
        elements = str_pair.strip('[]').split(', ')
        pair = tuple(elements)
        if embedding1 is None or embedding2 is None:
            print(f"Skipping pair {idx + 1}: Missing embedding(s).")
            rows.append([folder_name, pair[0], "Missing Embedding(s)", "N/A", "N/A", label])
            rows.append([folder_name, pair[1], "Missing Embedding(s)", "N/A", "N/A", label])
            continue

        # Compute distance and prediction
        distance = (embedding1 - embedding2).norm().item()
        prediction = 1 if distance < threshold else 0
        
        fv_prediction = "match" if prediction == 1 else "mismatch"
        ground_truth = "match" if label == 1 else "mismatch"
        print(f">>> label: {label}, ground_truth: {ground_truth}, prediction: {prediction}, distance: {distance}")

        # Determine conflict
        conflict = 1 if fv_prediction != ground_truth else 0

        rows.append([folder_name, pair[0], pair[1], fv_prediction, ground_truth, conflict, label])
        print(f"Folder: {folder_name}, Pairs: {pair[0]}, Pairs: {pair[1]}, FV Prediction: {fv_prediction}, Ground Truth: {ground_truth}, Conflict: {conflict}, label: {label}")

    # Create DataFrame
    columns = ["Folder Name", "Pair1", "Pair2", "FV Prediction", "Ground Truth", "Conflict", "Label"]
    df = pd.DataFrame(rows, columns=columns)

    # Remove duplicate rows based on "Pair National IDs"
    # df.drop_duplicates(subset=["Pairs"], inplace=True)

    # Save the final deduplicated DataFrame to CSV
    df.to_csv(output_csv, sep=";", index=False)
    print(f"\nFinal results logged to {output_csv}\n--- Evaluation Complete ---\n")



def process_person_folder(person_folder,
                          embeddings,
                          labels,
                          folder_names,
                          pairs_list,
                          model,
                          transform, device):
    """Extract embeddings and generate pairs for evaluation."""
    files = os.listdir(person_folder)
    files = sorted(files)

    # Separate files into reference (ref) and selfie categories
    ref_files = [f for f in files if '_1.jpg' in f]
    selfie_files = [f for f in files if '_2.jpg' in f]

    if len(ref_files) != 2 or len(selfie_files) != 2:
        logger.error(f"Invalid folder structure in {person_folder}: {files}")
        return [], [], [], []

    # Extract pairs
    pairs = [
        (os.path.join(person_folder, ref_files[0]), os.path.join(person_folder, selfie_files[1])),  # Pair 1
        (os.path.join(person_folder, ref_files[1]), os.path.join(person_folder, selfie_files[0])),  # Pair 2
        (os.path.join(person_folder, ref_files[0]), os.path.join(person_folder, selfie_files[0])),  # Pair 3
        (os.path.join(person_folder, ref_files[1]), os.path.join(person_folder, selfie_files[1]))   # Pair 4
    ]
    pair_labels = [0, 0, 1, 1]  # Pairs 1 and 2 are mismatches; Pairs 3 and 4 are matches


    # Generate embeddings and labels
    for (ref_path, selfie_path), label in zip(pairs, pair_labels):
        ref_embedding = get_face_embedding(ref_path, transform, model, device)
        selfie_embedding = get_face_embedding(selfie_path, transform, model, device)
        embeddings.append((ref_embedding, selfie_embedding))
        print(f"label: {label}")
        labels.append(label)
        folder_names.append(os.path.basename(person_folder))
        pairs_list.append(f"[{os.path.basename(ref_path)}, {os.path.basename(selfie_path)}]")

    return embeddings, labels, folder_names, pairs_list


if __name__ == "__main__":
    with open("config.yaml", "r") as config_file:
        config_dict = yaml.safe_load(config_file)
        config = Box(config_dict)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Available Device: {device}")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((config.train.transform.resize[0],
                           config.train.transform.resize[1])),
        transforms.ToTensor(),
        transforms.Normalize(config.train.transform.normalize.mean,
                             config.train.transform.normalize.std)
    ])

    checkpoint_path = os.path.join(config.inference.weight_dir, config.inference.checkpoint_name)
    model = load_model(device, checkpoint_path)


    test_dir = config.inference.test_dir

    embeddings, labels, folder_names, pairs_list = [], [], [], []
    embeddings, labels, folder_names, pairs_list = process_person_folder(test_dir,
                                                                         embeddings,
                                                                         labels,
                                                                         folder_names,
                                                                         pairs_list,
                                                                         model,
                                                                         transform,
                                                                         device)

    if all(list_ for list_ in [embeddings, labels, folder_names, pairs_list]):
        output_csv_path = os.path.join(test_dir, config.inference.output_csv)
        evaluate_and_log_pairs(embeddings,
                            labels,
                            folder_names,
                            pairs_list,
                            threshold=config.inference.opt_threshold,
                            output_csv=output_csv_path)
    else:
        logger.warning("can not evaluate images.")