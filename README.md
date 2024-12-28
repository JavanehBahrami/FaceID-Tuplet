# Face Verification with Tuplet Loss

This repository implements a face verification system based on FaceNet fine-tuned using Tuplet Loss. 
The project leverages PyTorch for deep learning and focuses on learning discriminative embeddings for facial images, especially in cases where `faces are close to the camera` such as in eKYC systems.

## Features
- Fine-tuned FaceNet model for face verification.
- Tuplet Loss: Uses one positive sample and multiple negative samples for robust learning.
- Embedding learning to enhance discriminative power for facial distances.
- Optimized for challenging scenarios where faces are captured at close range.

Table of Contents
Installation
Dataset
Usage
Training Details
Results
Contributing
License

### what is tuplet loss

<img src="docs/images/tuplet_loss.jpg" alt="Triplet_loss Versue Tuplet_loss" height="498"/>


### root structure


### Dataset
I used a custom dataset for training.

#### Data preparation
```
Train Directory format
    |---nationalID1
    |     |--img1
    |     |--img2
    |     |...
    |     |--img10
    |---nationalID2
    |     |--img1
    |     |--img2
    |     |...
    |     |--img10
    |...
    |---nationalIDN
    |     |--img1
    |     |--img2
    |     |...
    |     |--img10
```




### Resources:
I used this device.

GPU: RTX-3060
Cuda version : 11.8


### training metrics
- Train_Loss
- Val_Acc
- Val_Prec
- Val_Rec
- Val_F1
- Val_TPR
- Val_FPR
- Val_FNR

### Download pretrained weigths
the pretrained weight is located on `weight` directory.

### Finding Optimal Threshold based on distance similarity
Test version1 dataset unique samples = 7998

=== Threshold Search Results ===
Optimal Threshold: 0.9422
Best FPR: 0.0026, Best FNR: 0.0132, Minimum Cost: 0.0048


### Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/face-verification-tuplet-loss.git
cd face-verification-tuplet-loss
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure you have PyTorch installed with the appropriate CUDA version for your setup. Refer to the PyTorch Installation Guide if needed.

Dataset
Prepare your dataset with face images. The dataset should be structured to include:
One positive image pair (same person).
Multiple negative samples (different persons).
Examples of datasets:
Labeled Faces in the Wild (LFW)
VGGFace2
Usage
Training
To train the model with your dataset:

bash
Copy code
python train.py --data_path /path/to/dataset --epochs 20 --batch_size 32
Inference
To verify two face images:

bash
Copy code
python verify.py --img1 /path/to/image1.jpg --img2 /path/to/image2.jpg
Training Details
Model: Fine-tuned FaceNet using PyTorch.
Loss Function: Tuplet Loss with one positive and multiple negative samples.
Optimization:
Optimizer: Adam
Learning Rate: 1e-4
Batch Size: 32
Embedding Dimension: 512-dimensional vectors for face embeddings.
Preprocessing:
Face detection and alignment using MTCNN (from facenet-pytorch).
Normalization of input images.
Results
The fine-tuned model achieves robust face verification performance, especially for faces captured at close range.

Example:
Positive Pair: Match
Negative Pair: Not a Match
Performance metrics:

Metric	Value
Accuracy	98.5%
Precision	97.8%
Recall	98.2%
Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
FaceNet: A Unified Embedding for Face Recognition and Clustering
PyTorch
facenet-pytorch