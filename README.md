# Fake News Detection with NLP

A comparative text classification project that detects fake news using three Natural Language Processing approaches, from a simple word-count baseline up to a neural network. All three models use the same data split and the same evaluation, so their results can be compared fairly.

## Models

| # | Model | Text representation | Classifier |
|---|-------|---------------------|------------|
| 1 | Baseline | Bag of Words (`CountVectorizer`, 20,000 features) | Multinomial Naive Bayes |
| 2 | Statistical | TF-IDF with unigrams and bigrams (50,000 features), stopwords removed | Logistic Regression (class-balanced) |
| 3 | Embedding | Keras tokenizer, padded sequences, 128-dimension embeddings | Bidirectional LSTM with dropout and early stopping |

## Dataset

The [Fake News dataset from Kaggle](https://www.kaggle.com/datasets/saratchendra/fake-news) (`fake_train.csv`) with 20,761 articles and two columns: `text` and `label` (0 = fake, 1 = real). The classes are balanced (10,387 fake and 10,374 real). The data is split 80/20 into 16,608 training and 4,153 test articles, stratified by label.

The dataset is about 100 MB, so it is not stored in this repository.

## Results

Results on the 4,153-article test set, as recorded in the notebook:

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Baseline (BoW + MultinomialNB) | 0.8839 | 0.9146 | 0.8467 | 0.8794 |
| Statistical (TF-IDF + LogReg) | 0.9437 | 0.9440 | 0.9431 | 0.9436 |
| Embedding (BiLSTM) | 0.9420 | 0.9451 | 0.9383 | 0.9417 |

The notebook also includes a confusion matrix for each model, a bar chart comparing all of them, and a written discussion of the trade-offs between accuracy, cost and interpretability.

## Project Structure

| File | Purpose |
|------|---------|
| `fake_news_detection.ipynb` | The notebook: introduction, preprocessing, the three models, evaluation and conclusion |
| `fake_news_detection.pdf` | PDF export of the notebook with its outputs |
| `dataset/fake_train.csv` | The dataset (**not included**, see below) |

## Run Locally

You need Python 3 and Jupyter.

```bash
git clone https://github.com/ayeshaatif07/Fake-News-Detection.git
cd Fake-News-Detection
pip install scikit-learn tensorflow nltk pandas numpy matplotlib jupyter
mkdir dataset
```

Then:

1. Download `fake_train.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/saratchendra/fake-news) and put it in the `dataset/` folder.
2. Start Jupyter with `jupyter notebook fake_news_detection.ipynb`.
3. Run the cells from top to bottom. The NLTK data (`punkt` and `stopwords`) downloads automatically. Training the BiLSTM takes a few minutes on a CPU.

Running the notebook also saves the trained models (`.pkl` and `.keras` files) in the project folder.

## About This Notebook

This notebook was rebuilt from the PDF export of the original coursework. The code, text and outputs are taken from that PDF, and the charts are the images from it.

## Author and Credits

- **Author:** Ayesha Atif
- **Coursework:** Natural Language Processing
- Dataset: [Fake News](https://www.kaggle.com/datasets/saratchendra/fake-news) by S. Chendra on Kaggle.
- Built with [scikit-learn](https://scikit-learn.org/), [TensorFlow/Keras](https://www.tensorflow.org/), [NLTK](https://www.nltk.org/) and [pandas](https://pandas.pydata.org/).
