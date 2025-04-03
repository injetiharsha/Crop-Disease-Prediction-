# Plant Disease Detection

## Overview
This project implements a convolutional neural network (CNN) for detecting plant diseases from images of leaves. The model is trained using PyTorch and employs transfer learning and a custom CNN architecture. The dataset consists of images categorized into different classes of plant diseases.

## Features
- **Dataset Handling**: Loads and preprocesses images using PyTorch's `ImageFolder`.
- **Model Architecture**: Implements a CNN with multiple convolutional layers, batch normalization, dropout, and fully connected layers.
- **Training Pipeline**: Uses mini-batch gradient descent with Adam optimizer and cross-entropy loss.
- **Performance Evaluation**: Computes accuracy and F1-score on validation and test sets.
- **Model Saving & Loading**: Saves trained model weights and reloads for inference.
- **Inference**: Predicts disease class for a given image using a trained model.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```
2. Install dependencies:
   ```sh
   pip install torch torchvision numpy pandas matplotlib tqdm sklearn pillow
   ```

## Dataset Preparation
- The dataset should be a directory structured as follows:
  ```
  Plant_leave_diseases_dataset_without_augmentation/
  ├── Class1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── Class2/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── ...
  ```
- The dataset is automatically extracted if not found in the specified path.

## Training the Model
Run the following command to start training:
```sh
python train.py
```
- Training runs for 10 epochs by default.
- Progress is displayed using `tqdm`.
- The trained model is saved as `plant_disease_model.pth`.

## Evaluating the Model
Run the evaluation script:
```sh
python evaluate.py
```
- Computes accuracy and F1-score for train, validation, and test sets.

## Running Inference
To classify a new leaf image:
```sh
python predict.py --image_path path/to/image.jpg
```
- Loads the trained model and applies transformations to the input image.
- Prints the predicted class.

## Results
- Training and validation accuracy are plotted after training.
- The model achieves high classification accuracy on test images.

## Dependencies
- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- tqdm

## Acknowledgments
- The dataset was obtained from a publicly available plant disease dataset.
- The model was trained using PyTorch and torchvision.

## License
This project is licensed under the MIT License.

