# Classification of Pulsar Candidates Using a Multilayer Perceptron (MLP) Neural Network

This repository contains a project focused on building a Multilayer Perceptron (MLP) neural network for the classification of pulsar star candidates. The goal of the project is to identify whether given astronomical measurements correspond to a pulsar or a non-pulsar (noise), using Python and TensorFlow for modeling and analysis.

## Project Overview

Pulsars are a type of neutron star that emit beams of radiation, detectable as pulses due to their rotation. This project leverages machine learning techniques to classify pulsar candidates using a dataset of various radio frequency measurements.

### Features of the Project
- Data Preprocessing: Cleaning, scaling, and analyzing the dataset.
- Model Development: Building a MLP neural network using TensorFlow/Keras.
- Evaluation Metrics: Analyzing accuracy, precision, recall, F1 score, and visualizing confusion matrices.
- Visualization: Graphing the model architecture and training history.

### Dataset

The dataset used in this project is publicly available and contains radio frequency measurements from pulsar candidates. It has several features, including the mean, standard deviation, skewness, and kurtosis of different profiles.

### Project Structure
- `ANN_MLP_Pulsar.ipynb`: The main Jupyter notebook containing all steps of the project, including data preprocessing, model building, and evaluation.
- `README.md`: Project description, including installation and usage instructions.
- `.gitignore`: Files and folders to be ignored by Git.
- `LICENSE`: Licensing information for using this code.

### Requirements

To run the project locally, you will need Python 3.9.19 and the following Python libraries:
- `tensorflow`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`

To install the required dependencies, run:

```
pip install -r requirements.txt
```

### Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/pulsar-classification-mlp.git
```
2. Install the dependencies as mentioned above.
3. Open `ANN_MLP_Pulsar.ipynb` using Jupyter Notebook or any compatible IDE.

### Usage
- Run the notebook to execute the full process of loading, preprocessing, training, and evaluating the MLP model.
- Visualize the model training history and confusion matrix for detailed analysis of model performance.

### Model Summary
The MLP model consists of:
- Input Layer: Accepts radio frequency measurements.
- Hidden Layers: Includes ReLU activation functions with Dropout regularization.
- Output Layer: Uses sigmoid activation for binary classification.

### Results
The model's performance metrics include accuracy, precision, recall, and F1 score, which are computed and visualized in the Jupyter notebook.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contributing
Contributions are welcome! If you have suggestions for improvements, feel free to fork the repository and submit a pull request.

### Acknowledgments
- Dataset: The dataset used for this project is available from the UCI Machine Learning Repository.
- Libraries: TensorFlow, Keras, scikit-learn, pandas, matplotlib, and seaborn were used for building and evaluating the model.

### Author
Developed by L. Felipe Castañeda G.
