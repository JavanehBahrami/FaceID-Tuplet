import os
import torch
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import yaml
import pdb

from box import Box
from loguru import logger
from facenet_pytorch import InceptionResnetV1
from PIL import Image


def load_model(device, checkpoint_path):

    # Load the pre-trained model without classification layer
    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    logger.info("pre-trained model is loaded successfully without classifier layer.")

    # Load the fine-tuned checkpoint
    # checkpoint_name = 'checkpoint_epoch56_loss0.0308_valAcc1.0000_valTPR1.0000_valFPR0.0000_valFNR0.0000.pth'
    # checkpoint_path = f'/hdd_dr/dataset/VAP/project/FV/replace_FV_model/facenet_pytorch/examples2/chkp/test29_tuplet_loss/{checkpoint_name}'
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    logger.info("Checkpoint is loaded successfully.")
    print("----------------------------------------------------------------------------------")

    return model



# Function to extract face embeddings
def get_face_embedding(image_path):
    """Extracts the embedding of a given image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((160, 160))  # Resize image to 160x160
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    with torch.no_grad():
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0  # Normalize image
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
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

    # Return metrics
    return {
        "FP": fp,
        "FN": fn,
        "FPR": fpr,
        "FNR": fnr,
        "TP": tp,
        "TN": tn
    }



def evaluate_and_log_pairs(embeddings, labels, folder_names, pairs_list, threshold=0.96, output_csv="fv_results.csv"):
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
    
    # Temporary storage for rows
    rows = []

    for idx, ((embedding1, embedding2), label, folder_name, pair) in enumerate(zip(embeddings, labels, folder_names, pairs_list)):
        if embedding1 is None or embedding2 is None:
            print(f"Skipping pair {idx + 1}: Missing embedding(s).")
            rows.append([folder_name, pair, "Missing Embedding(s)", "N/A", "N/A", label])
            continue

        # Compute distance and prediction
        distance = (embedding1 - embedding2).norm().item()
        prediction = 1 if distance < threshold else 0
        
        fv_result = "match" if prediction == 1 else "mismatch"
        ground_truth = "match" if label == 1 else "mismatch"

        # Determine conflict
        conflict = 1 if fv_result != ground_truth else 0

        # Store the row
        rows.append([folder_name, pair, fv_result, ground_truth, conflict, label])
        print(f"Folder: {folder_name}, Pair: {pair}, FV Result: {fv_result}, Ground Truth: {ground_truth}, Conflict: {conflict}")

    # Create DataFrame
    columns = ["Folder Name", "Pair National IDs", "FV Result", "Ground Truth", "Conflict", "Label"]
    df = pd.DataFrame(rows, columns=columns)

    # Remove duplicate rows based on "Pair National IDs"
    df.drop_duplicates(subset=["Pair National IDs"], inplace=True)

    # Save the final deduplicated DataFrame to CSV
    df.to_csv(output_csv, sep=";", index=False)
    print(f"\nFinal results logged to {output_csv}\n--- Evaluation Complete ---\n")

    # Compute metrics on deduplicated data
    predictions = [1 if row["FV Result"] == "match" else 0 for _, row in df.iterrows()]
    labels = df["Label"].tolist()

    metrics = compute_inference_metrics(predictions, labels)
    print("\n--- Inference Metrics After Removing Duplicates ---")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")
    print(f"False Positive Rate (FPR): {metrics['FPR']:.4f}")
    print(f"False Negative Rate (FNR): {metrics['FNR']:.4f}")
    print(f"True Positives (TP): {metrics['TP']}")
    print(f"True Negatives (TN): {metrics['TN']}")

    # Compute distribution of matching and non-matching pairs
    num_matching = labels.count(1)
    num_non_matching = labels.count(0)
    total_pairs = len(labels)

    print(f"\n--- Pair Distribution ---")
    print(f"Matching Pairs: {num_matching} ({num_matching / total_pairs * 100:.2f}%)")
    print(f"Non-Matching Pairs: {num_non_matching} ({num_non_matching / total_pairs * 100:.2f}%)")

    # Visualize distribution
    # Save distribution as a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie([num_matching, num_non_matching], labels=["Matching", "Non-Matching"], autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#F44336"])
    plt.title("Pair Distribution")
    output_image = "pair_distribution.png"
    plt.savefig(output_image)  # Save the chart as a PNG file
    plt.close()  # Close the plot to free memory
    print(f"Pair distribution pie chart saved to {output_image}.")


# Process a folder to generate pairs and collect embeddings
def process_person_folder(person_folder, embeddings, labels, folder_names, pairs_list):
    """Extract embeddings and generate pairs for evaluation."""
    files = os.listdir(person_folder)
    files = sorted(files)

    # Separate files into reference (ref) and selfie categories
    ref_files = [f for f in files if '_1.jpg' in f]
    selfie_files = [f for f in files if '_2.jpg' in f]
    pdb.set_trace()

    if len(ref_files) != 2 or len(selfie_files) != 2:
        print(f"Invalid folder structure in {person_folder}: {files}")
        return

    # Extract pairs
    pairs = [
        (os.path.join(person_folder, ref_files[0]), os.path.join(person_folder, selfie_files[1])),  # Pair 1
        (os.path.join(person_folder, ref_files[1]), os.path.join(person_folder, selfie_files[0])),  # Pair 2
        (os.path.join(person_folder, ref_files[0]), os.path.join(person_folder, selfie_files[0])),  # Pair 3
        (os.path.join(person_folder, ref_files[1]), os.path.join(person_folder, selfie_files[1]))   # Pair 4
    ]
    pair_labels = [0, 0, 1, 1]  # Pairs 1 and 2 are mismatches; Pairs 3 and 4 are matches
    pdb.set_trace()

    # Generate embeddings and labels
    for (ref_path, selfie_path), label in zip(pairs, pair_labels):
        ref_embedding = get_face_embedding(ref_path)
        selfie_embedding = get_face_embedding(selfie_path)
        embeddings.append((ref_embedding, selfie_embedding))
        labels.append(label)
        folder_names.append(os.path.basename(person_folder))
        pairs_list.append(f"[{os.path.basename(ref_path)}, {os.path.basename(selfie_path)}]")


def process_all_folders(root_dir, output_csv="fv_results.csv"):
    embeddings, labels, folder_names, pairs_list = [], [], [], []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        # pdb.set_trace()
        if os.path.isdir(folder_path):
            process_person_folder(folder_path, embeddings, labels, folder_names, pairs_list)

    # Evaluate all pairs and log results
    evaluate_and_log_pairs(embeddings, labels, folder_names, pairs_list, threshold=0.96, output_csv=output_csv)


if __name__ == "__main__":
    with open("config.yaml", "r") as config_file:
        config_dict = yaml.safe_load(config_file)
        config = Box(config_dict)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Available Device: {device}")

    checkpoint_path = os.path.join(config.inference.weight_dir, config.inference.checkpoint_name)
    load_model(device, checkpoint_path)

    # Example usage
    test_dir = config.inference.test_dir
    process_all_folders(test_dir, output_csv="fv_results_25_day.csv")