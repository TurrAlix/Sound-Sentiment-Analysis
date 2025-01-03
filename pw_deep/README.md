# Sound Sentiment Analysis

This repository contains the implementation of a **deep learning project** for sound sentiment analysis. The objective is to predict the sentiment conveyed in audio clips using state-of-the-art techniques in machine learning and deep learning.

## Project Overview

This project is based on the [AI Blitz: Sound Sentiment Prediction](https://www.aicrowd.com/blitz/puzzles/sound-sentiment-prediction). The aim is to develop a model that accurately classifies audio samples into predefined sentiment categories.

### Dataset

The dataset for this project is hosted on Google Drive and can be accessed [here](https://drive.google.com/drive/folders/1n7ja9PlDLfnPFmU3AMjo8lca5zy3iP6t). It consists of labeled audio files representing different sentiments. The dataset is divided into training, validation and testing sets for model development and evaluation.

### Objectives

1. Preprocess the audio data for feature extraction.
2. Build and fine-tune a deep learning model for sentiment classification. Combining 2 approaches: text and images
3. Evaluate the model's performance on test data.

## Getting Started

### Prerequisites

- Python 3.10+
- Recommended libraries: `numpy`, `pandas`, `tensorflow`/`pytorch`, `librosa`, `scikit-learn`, `matplotlib`, `seaborn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sound-sentiment-analysis.git
   ```
2. Install the required dependencies:
   ```bash
   poetry install
   poetry shell
   ```

### Usage

1. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1n7ja9PlDLfnPFmU3AMjo8lca5zy3iP6t) and place it in the `data` directory.
2. Explore the data using the notebook, and Train the models.
4. Evaluate the model and visualize results in the `results` directory.

## Contributions

Contributions to the project are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [AI Blitz: Sound Sentiment Prediction](https://www.aicrowd.com/blitz/puzzles/sound-sentiment-prediction) for providing the inspiration and dataset for this project.
- Open-source libraries and tools that made this project possible.