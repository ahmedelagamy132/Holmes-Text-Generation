
# Holmes-Text-Generation

**Sherlock Holmes Text Generation with GPT-2 and LSTM**  
This project focuses on generating text inspired by the Sherlock Holmes stories using two different approaches:

1. Fine-tuning GPT-2 for text generation.  
2. Training an LSTM model from scratch for text generation.  

The project includes preprocessing, model training, evaluation, and text generation capabilities. The dataset used is the complete text of Arthur Conan Doyle's Sherlock Holmes stories, which is in the public domain.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Training](#model-training)  
- [Evaluation](#evaluation)  
- [Contributing](#contributing)  
- [License](#license)  
- [Dataset Documentation](#dataset-documentation)  
- [Code Structure](#code-structure)  
- [Future Improvements](#future-improvements)  
- [Conclusion](#conclusion)  

---

## Project Overview

The project is divided into two main parts:  

- **GPT-2 Fine-tuning**: Using the Hugging Face transformers library, we fine-tune the GPT-2 model on the Sherlock Holmes dataset to generate text in the style of Arthur Conan Doyle.  
- **LSTM Model**: We build and train an LSTM model from scratch using TensorFlow/Keras to generate text sequences.

### Key Features:
- Preprocessing of the Sherlock Holmes text dataset.  
- Fine-tuning GPT-2 for text generation.  
- Training an LSTM model for text generation.  
- Evaluation using metrics like Top-K Accuracy, BLEU Score, and Perplexity.  
- Text generation using both models.  

---

## Installation

To run this project, you need to install the required dependencies:  
```bash
pip install -r requirements.txt
```

### Requirements:
- Python 3.8+  
- TensorFlow 2.x  
- Hugging Face Transformers  
- NLTK  
- NumPy  
- Pandas  
- Matplotlib  
- WordCloud  
- NetworkX  

---

## Usage

### 1. GPT-2 Fine-tuning
To fine-tune the GPT-2 model, run the `app.py` script:
```bash
python app.py
```
This script will:  
- Load the Sherlock Holmes dataset.  
- Preprocess the text.  
- Fine-tune the GPT-2 model.  
- Save the fine-tuned model and tokenizer.  
- Generate text using the fine-tuned model.  

### 2. LSTM Model
To train the LSTM model, run the `from_scratch.ipynb` notebook. It includes:  
- Data preprocessing.  
- Tokenization and sequence generation.  
- Model training and evaluation.  
- Text generation using the trained LSTM model.  

---

## Model Training

### GPT-2 Fine-tuning
- Tokenization of the dataset.  
- Training for 3 epochs with a learning rate of `5e-5`.  
- Evaluation using validation loss.  

### LSTM Model
- Tokenization and sequence generation.  
- Training for 100 epochs with Adam optimizer.  
- Evaluation using Top-K Accuracy, BLEU Score, and Perplexity.  

---

## Evaluation

### Metrics:
- **Top-K Accuracy**: Measures the accuracy of the model's top-K predictions.  
- **BLEU Score**: Evaluates the quality of generated text by comparing it to the reference text.  
- **Perplexity**: Measures how well the model predicts the next word in a sequence.  

---



## Contributing

Contributions are welcome! If you'd like to contribute:  
1. Fork the repository.  
2. Create a new branch for your feature or bug fix.  
3. Commit your changes.  
4. Submit a pull request.  

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.  

---

## Dataset Documentation

### Reference:
The dataset, titled **"Sherlock Holmes Stories"**, is from a public domain source and provides the full text of Arthur Conan Doyle's Sherlock Holmes stories.  

---

## Code Structure

### Files:
- `app.py`: Script for fine-tuning GPT-2 and generating text.  
- `from_scratch.ipynb`: Jupyter notebook for training the LSTM model from scratch.  
- `combined_sherlock_holmes.txt`: Dataset containing the Sherlock Holmes stories.  

### Key Functions:
- **Preprocessing**: Cleans and tokenizes the dataset.  
- **Model Training**: Fine-tunes GPT-2 and trains the LSTM model.  
- **Evaluation**: Calculates metrics like Top-K Accuracy, BLEU Score, and Perplexity.  

---

## Future Improvements

- **Training Data**: Use additional high-quality text data to improve grammar and context understanding.  
- **Model Architecture**: Add layers or fine-tune hyperparameters for better predictions.  
- **Beam Search**: Implement beam search to enhance sentence coherence.  

---

## Conclusion

The project demonstrates basic text generation capabilities using both GPT-2 and LSTM models. While results are promising, further refinement is needed to improve fluency and grammatical correctness. Contributions and suggestions are welcome!  
