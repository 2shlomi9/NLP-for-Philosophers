# Philosophy Text Classification Project, using Deep Learning and NLP methods

## Overview
This project focuses on classifying sentences from philosophical texts into one of four major philosophical schools: Aristotle, German Idealism, Plato, and Continental Philosophy. The dataset contains over 150,000 sentences extracted from 51 philosophical texts.

We implemented various machine learning and deep learning models, ranging from Logistic Regression to BERT, to analyze and classify these texts effectively.

---

## Project Goals
1. Develop and evaluate machine learning and deep learning models for philosophical text classification.
2. Explore text processing methods to enhance feature representation.
3. Compare the effectiveness of different model architectures.
4. Gain insights into the conceptual and ideological distinctions reflected in the dataset.

---

## Dataset
- The dataset includes 150,000 sentences divided into four philosophical schools:
  - Aristotle: 48,778 sentences
  - German Idealism: 42,135 sentences
  - Plato: 38,365 sentences
  - Continental: 33,778 sentences
- Extensive preprocessing was performed to ensure high-quality data for modeling.

---

## Models Implemented

### Logistic Regression (LR)
- Implemented in the `LR` directory.
- Features:
  - Conversion methods: Bag-Of-Words, TF-IDF, Word2Vec.
  - Regularization options: Lasso, Ridge.
  - Optimization methods: SGD, ADAM, LBFGS.
- Results:
  - Accuracy: **84%**
- Graphs:
  ![Add NN Training Graph Here]
  ![Add NN Validation Graph Here]
- To run:
  ```bash
  python main_lr.py
  ```
  Options in `main_lr.py`:
  - `choosen_convert_data`: Choose a conversion method (1 - Bag-Of-Words, 2 - TF-IDF, 3 - Word2Vec).
  - `choosen_training_model`: Choose training method (0 - without cross-validation, 1 - with cross-validation).
  - `choosen_regularization`: Choose regularization (0 - without, 1 - Lasso, 2 - Ridge).
  - `choosen_optimization`: Choose optimization (1 - SGD, 2 - ADAM, 3 - LBFGS).

---

### Fully Connected Neural Network (NN)
- Implemented in the `NN` directory.
- Features:
  - Two hidden layers (1024 and 512 neurons).
  - Techniques: Batch Normalization, Dropout, Learning Rate Scheduler.
  - Optimizer: Adam.
- Results:
  - Accuracy: **84%**
- Graphs:
  ![Add NN Training Graph Here]
  ![Add NN Validation Graph Here]
- To run:
  ```bash
  python main_nn.py
  ```

---

### Recurrent Neural Network (RNN)
- Implemented in the `RNN` directory.
- Features:
  - Bidirectional LSTM with Attention.
  - Components:
    - Embedding Layer: Converts input sequences into dense vectors.
    - Bidirectional LSTM: Processes sequences in both directions.
    - Attention Layer: Focuses on the most relevant parts of the input.
  - Optimizations: Cyclic Learning Rate Scheduler, Dropout.
- Results:
  - Accuracy: **85%**
- Graphs:
  ![Add RNN Training Graph Here]
  ![Add RNN Validation Graph Here]
- To run:
  ```bash
  python main_rnn.py
  ```

---

### BERT
- Implemented in the `Bert` directory.
- Features:
  - Pre-trained BERT model fine-tuned for text classification.
  - Optimizer: AdamW.
  - Learning Rate Scheduler: Linear schedule with warmup.
- Results:
  - Accuracy: **89%**
- Graphs:
  ![Add BERT Training Graph Here]
  ![Add BERT Validation Graph Here]
- To run:
  ```bash
  python main_bert.py
  ```

---

## Directory Structure
```
project/
├── LR/      # Logistic Regression implementation
├── NN/      # Fully Connected Neural Network implementation
├── RNN/     # Recurrent Neural Network implementation
├── Bert/    # BERT implementation
├── Models/  # Trained models saved here
```

---

## Text Processing
Text preprocessing steps included:
1. **Removing Irrelevant Words:** Stopwords like "is," "she," etc., were removed.
2. **Converting to Lowercase:** Ensures uniformity.
3. **Removing Punctuation Marks.**
4. **Lemmatization:** Converts words to their base forms.
5. **Vectorization:**
   - TF-IDF: Calculates importance of words.
   - Bag-Of-Words: Represents text as word frequency arrays.
   - Word2Vec: Generates semantic vectors for words.

---

## Results Summary
- **Logistic Regression:** Best accuracy: **81%** (Word2Vec + Ridge Regularization).
- **NN:** Best accuracy: **84%.**
- **RNN:** Best accuracy: **85%.**
- **BERT:** Best accuracy: **89%.**

### Performance Progression
![Add Accuracy Graph Here]

### Final Results:
![Add Final Results Here]

---

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the desired model:
   - Logistic Regression:
     ```bash
     python main_lr.py
     ```
   - Neural Network:
     ```bash
     python main_nn.py
     ```
   - RNN:
     ```bash
     python main_rnn.py
     ```
   - BERT:
     ```bash
     python main_bert.py
     ```

---

## Authors
- **Matan Blaich**
- **Shlomi Zecharia**
- **Abedallah Zoabi**

---

## Notes
For visualization of training metrics and results, refer to the appropriate sections in the outputs or add graphs/images to the placeholders in this README.
