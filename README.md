# LSTMAttention-IMDB

# LSTM with Attention for IMDB Reviews

This project implements a sentiment analysis model using an LSTM with an attention mechanism to classify movie reviews from the IMDB dataset. 

The model is built with PyTorch and processes text data using tokenization and padding techniques.

**Tools and Libraries used:**
* PyTorch
* Torchtext
* Spacy
* NumPy
* Pandas
* Scikit-learn



## Table of Contents

- [Project Overview](#project-overview)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)


## Project Overview

The goal of this project is to build a text classification model that can determine the sentiment of IMDB movie reviews. The model utilizes an LSTM (Long Short-Term Memory) network with an attention mechanism to better capture relevant information in the text. 

### Features

- **LSTM with Attention**: A neural network model that combines LSTM with attention to improve performance on text classification tasks.
- **Data Processing**: Includes text cleaning, tokenization, and padding to prepare data for model training.
- **Training and Evaluation**: Scripts to train the model and evaluate its performance using accuracy and loss metrics.

## Installation Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/LSTMAttention-IMDB.git
    cd LSTMAttention-IMDB
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install dependencies:**

    Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Spacy model:**

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

1. **Prepare the data and run the training:**

    ```bash
    python main.py
    ```

    This command starts the training process and evaluates the model on the test dataset.



## Code Structure

- **`main.py`**: Runs the training and evaluation process. It loads the dataset, initializes the model, trains it, and evaluates it.
- **`model.py`**: Defines the model architecture. Contains the LSTM with Attention model definition.
- **`data_loader.py`**: Manages data loading and preprocessing. Conatins the functions for data processing including text tokenization, padding, and data loading.
- **`train.py`**: Contains functions for training and evaluation.

## Dependencies

- **PyTorch**: For building and training the model.
- **Torchtext**: For text processing and data loading.
- **Spacy**: For text tokenization.

