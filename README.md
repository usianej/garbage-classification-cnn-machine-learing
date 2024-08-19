# Garbage Classification with TensorFlow and Keras

Welcome to my Garbage Classification project! This repository demonstrates my skills in Python and machine learning through a practical application of classifying waste images using deep learning techniques.

## Project Overview

The goal of this project is to develop a machine-learning model that can accurately classify images of different types of garbage. Using TensorFlow and Keras, I've built a Convolutional Neural Network (CNN) to identify and categorize waste items.

## Key Features

- **Data Handling**: Utilized Python libraries to manage and preprocess image data. Employed `splitfolders` for organizing datasets and `pandas` for efficient data manipulation.
- **Image Processing**: Implemented data augmentation and normalization techniques to enhance model performance, including resizing images for uniform input.
- **Model Building**: Designed and trained a CNN with layers such as Convolution2D, MaxPooling2D, Flatten, and Dense to achieve robust image classification.
- **Training and Optimization**: Applied various training strategies and optimizers. Used `EarlyStopping` to prevent overfitting and improve model generalization.
- **Evaluation and Visualization**: Assessed model performance using confusion matrices and accuracy scores. Visualized results with Matplotlib to understand training progress and accuracy.

## Getting Started

To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/usianej/garbage-classification-cnn-machine-learing
   cd garbage-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:
   Download and prepare your dataset according to the instructions in `data/README.md`.

4. **Run the Model**:
   Execute the training script:
   ```bash
   python train_model.py
   ```

5. **Evaluate and Predict**:
   Use `evaluate_model.py` to assess model performance and `predict.py` for making predictions on new images.

## Project Structure

- `data/`: Contains dataset and preprocessing instructions.
- `src/`: Source code including model architecture, training, and evaluation scripts.
- `notebooks/`: Jupyter notebooks for exploratory analysis and visualization.
- `requirements.txt`: List of Python dependencies.

## Contributing

Feel free to fork this repository and submit pull requests. Contributions, issues, and feedback are welcome!

## License

This project is licensed under the Apache-2.0 license. See `LICENSE` for details.
