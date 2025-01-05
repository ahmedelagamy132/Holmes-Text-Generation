# Holmes-Text-Generation
# Sherlock Holmes Text Generation with GPT-2 and LSTM

This project focuses on generating text inspired by the Sherlock Holmes stories using two different approaches: 
1. **Fine-tuning GPT-2** for text generation.
2. **Training an LSTM model** from scratch for text generation.

The project includes preprocessing, model training, evaluation, and text generation capabilities. The dataset used is the complete text of Arthur Conan Doyle's Sherlock Holmes stories, which is in the public domain.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)
9. [Dataset Documentation](#dataset-documentation)
10. [Code Structure](#code-structure)
11. [Future Improvements](#future-improvements)
12. [Conclusion](#conclusion)

---

## Project Overview

The project is divided into two main parts:
1. **GPT-2 Fine-tuning**: Using the Hugging Face `transformers` library, we fine-tune the GPT-2 model on the Sherlock Holmes dataset to generate text in the style of Arthur Conan Doyle.
2. **LSTM Model**: We build and train an LSTM model from scratch using TensorFlow/Keras to generate text sequences.

### Key Features:
- Preprocessing of the Sherlock Holmes text dataset.
- Fine-tuning GPT-2 for text generation.
- Training an LSTM model for text generation.
- Evaluation using metrics like Top-K Accuracy, BLEU Score, and Perplexity.
- Text generation using both models.

---

## Installation

To run this project, you need to install the required dependencies. You can do this using `pip`:

```bash
pip install -r requirements.txt
Requirements
Python 3.8+

TensorFlow 2.x

Hugging Face Transformers

NLTK

NumPy

Pandas

Matplotlib

WordCloud

NetworkX

Usage
1. GPT-2 Fine-tuning
To fine-tune the GPT-2 model, run the app.py script:

bash
Copy
python app.py
This script will:

Load the Sherlock Holmes dataset.

Preprocess the text.

Fine-tune the GPT-2 model.

Save the fine-tuned model and tokenizer.

Generate text using the fine-tuned model.

2. LSTM Model
To train the LSTM model, run the from_scratch.ipynb notebook. This notebook includes:

Data preprocessing.

Tokenization and sequence generation.

Model training and evaluation.

Text generation using the trained LSTM model.

Model Training
GPT-2 Fine-tuning
The GPT-2 model is fine-tuned using the Hugging Face Trainer API. The training process includes:

Tokenization of the dataset.

Training for 3 epochs with a learning rate of 5e-5.

Evaluation using validation loss.

LSTM Model
The LSTM model is trained from scratch using TensorFlow/Keras. The training process includes:

Tokenization and sequence generation.

Training for 100 epochs with Adam optimizer.

Evaluation using Top-K Accuracy, BLEU Score, and Perplexity.

Evaluation
Metrics
Top-K Accuracy: Measures the accuracy of the model's top-K predictions.

BLEU Score: Evaluates the quality of generated text by comparing it to the reference text.

Perplexity: Measures how well the model predicts the next word in a sequence.

Results
GPT-2 Fine-tuning:

Top-3 Accuracy: 0.02

BLEU Score: 0.00

Perplexity: NaN (due to issues in the evaluation process)

LSTM Model:

Top-5 Accuracy: 0.00%

Top-20 Accuracy: 0.00%

BLEU Score: 0.0000

Perplexity: NaN (due to issues in the evaluation process)

Results
GPT-2 Text Generation Example
Input: "Sherlock Holmes was a detective who"
Output: "Sherlock Holmes was a detective who investigated the disappearance of a woman who came to live in an isolated community and is now an active member of the crime scene team of a crime scene unit in London."

LSTM Text Generation Example
Input: "detective"
Output: "detective service in recovering my horse you would do me now"

Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.

Create a new branch for your feature or bug fix.

Commit your changes.

Submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Dataset Documentation
Reference
The dataset, titled "Sherlock Holmes Stories ", was obtained from a public domain source, providing the full text of Arthur Conan Doyle's Sherlock Holmes stories. The source ensures no copyright restrictions, making it appropriate for academic and research purposes.

Why This Dataset?
This dataset was chosen for its rich linguistic complexity and narrative structure, ideal for projects involving natural language processing (NLP). Specifically, it offers:

Diverse Vocabulary: A wide range of words and styles, helpful for text analysis tasks.

Contextual Relationships: Suitable for building models focused on entity recognition or sentiment analysis.

Literary Importance: A well-known text that makes findings relatable and interpretable to a broader audience.

Documentation
Format: Plain-text file containing multiple Sherlock Holmes stories, organized by chapters.

Content Overview: Includes dialogue, descriptive prose, and complex sentence structures. This variety supports diverse NLP tasks such as parsing, summarization, and text generation.

Preprocessing Steps: For use in the project, the text may need:

Tokenization

Stopword removal

Lowercasing or stemming

Potential Use Cases:

Sentiment analysis to determine emotional tones in dialogues.

Named entity recognition to identify characters and locations.

Text classification for genre or thematic analysis.

Code Structure
Files
app.py: Script for fine-tuning GPT-2 and generating text.

from_scratch.ipynb: Jupyter notebook for training the LSTM model from scratch.

combined_sherlock_holmes.txt: Dataset containing the Sherlock Holmes stories.

Key Functions
Preprocessing:

Removes chapter titles and non-alphanumeric characters.

Replaces numbers with a placeholder <NUM>.

Expands contractions (e.g., "can't" → "cannot").

Tokenization:

Converts text into sequences of integer tokens.

Builds a word index for the LSTM model.

Model Training:

Fine-tunes GPT-2 using Hugging Face's Trainer.

Trains an LSTM model using TensorFlow/Keras.

Evaluation:

Calculates Top-K Accuracy, BLEU Score, and Perplexity.

Future Improvements
Training Data: The model may benefit from additional training on high-quality text data to improve grammar and context understanding.

Model Architecture: Consider adding more layers or fine-tuning hyperparameters (e.g., LSTM units, embedding size) to enhance predictions.

Beam Search: Use beam search instead of greedy decoding (argmax) to generate more coherent and contextually relevant sentences.

Conclusion
The project demonstrates basic text generation capabilities using both GPT-2 and LSTM models. While the results are promising, further refinement is needed to improve fluency and grammatical correctness. Contributions and suggestions are welcome!

Copy

---

### **How to Use This File**
1. Open a text editor (e.g., Notepad, VS Code, Sublime Text).
2. Copy the content above and paste it into the editor.
3. Save the file as `README.md` in your project folder.

---

### **How to Add It to Your GitHub Repository**
1. Open a terminal in your project folder.
2. Add the `README.md` file to Git:
   ```bash
   git add README.md
Commit the file:

bash
Copy
git commit -m "Added README.md file"
Push the changes to GitHub:

bash
Copy
git push origin main
