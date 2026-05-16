# Fake-Review-Classification-System

## 1. Project Overview

This project is a **Fake Review Classification System** that detects whether a review is **Fake** or **Genuine** using five machine learning models.

The system contains two main parts:

1. **Model Training Notebook** (`Fake_Review_Detection.ipynb`)
2. **Streamlit Web Application** (`app.py`)

The application allows users to enter a review and receive predictions from all trained models on one page.

---

## 2. Dataset

The dataset used in this project was obtained from Kaggle:
https://www.kaggle.com/datasets/muqaddasejaz/fake-reviews-dataset

Dataset name: **Fake Reviews Dataset**

This dataset contains review text with labels used for fake/genuine review classification.

---

## 3. Project Structure

```text id="u6k2pz"
project_folder/
│── models/
│   ├── svc.joblib
│   ├── logistic_reg.joblib
│   ├── multinomial.joblib
│   ├── random_forest.joblib
│   ├── sgd_classifier.joblib
│   └── vectorizer.joblib
│
│── notebook/
│   └── Fake_Review_Detection.ipynb
│
│── app.py
│── requirements.txt
```

### Folder Description

* **models/** → Contains all trained machine learning models and TF-IDF vectorizer
* **notebook/** → Contains the training notebook
* **app.py** → Streamlit web application
* **requirements.txt** → Python dependencies required to run the project

---

## 4. Files Included

### `Fake_Review_Detection.ipynb`

This notebook is used for:

* Loading and preparing the dataset
* Text preprocessing
* TF-IDF feature extraction
* Training five machine learning models
* Saving trained models and vectorizer using Joblib

### `app.py`

This is the Streamlit user interface used to:

* Accept user review text
* Apply the same preprocessing used during training
* Transform text using saved TF-IDF vectorizer
* Run predictions using all five models
* Display individual model predictions
* Display final majority voting result

---

## 5. Machine Learning Models Used

The system uses the following classifiers:

1. SVC
2. Logistic Regression
3. Multinomial Naive Bayes
4. Random Forest
5. SGD Classifier

Each model was trained separately and saved as a `.joblib` file.

---

## 6. Text Preprocessing

The same preprocessing pipeline is used in training and deployment.

Steps applied:

1. Convert text to lowercase
2. Remove URLs
3. Remove HTML tags using BeautifulSoup
4. Remove extra spaces
5. Tokenize using spaCy
6. Remove stopwords
7. Remove punctuation
8. Apply lemmatization
9. Join cleaned tokens into final processed text

---

## 7. Feature Extraction

TF-IDF Vectorizer is used to convert processed text into numerical features.

Configuration used:

* `max_features = 5000`
* `ngram_range = (1,2)`
* `min_df = 2`
* `max_df = 0.95`

This vectorizer was trained on the dataset and saved for reuse in the Streamlit application.

---

## 8. Saved Files

The application loads the following saved files from the `models/` folder:

* `svc.joblib`
* `logistic_reg.joblib`
* `multinomial.joblib`
* `random_forest.joblib`
* `sgd_classifier.joblib`
* `vectorizer.joblib`

---

## 9. How the Application Works

### User Workflow

1. Open the Streamlit application
2. Enter a review in the text box
3. Click **Process**
4. The system preprocesses the review
5. TF-IDF converts text into numerical features
6. All five models generate predictions
7. Results are shown in a table
8. Final decision is calculated using majority voting

---

## 10. Prediction Labels

The application uses:

* `0 = Fake`
* `1 = Genuine`

Displayed labels:

* Fake Review
* Genuine Review

---

## 11. Majority Voting Logic

After all models predict:

* If most models predict **Fake**, final result is Fake Review
* Otherwise, final result is Genuine Review

---

## 12. User Interface Features

Implemented in `app.py`:

* Wide page layout
* Review text input area
* One **Process** button
* Table of model predictions
* Final voting result
* Footer description

---

## 13. Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* Joblib
* spaCy
* BeautifulSoup

---

## 14. Running the Project

Install dependencies:

```bash id="t76lw5"
pip install -r requirements.txt
```

Run Streamlit:

```bash id="t9b2i0"
streamlit run app.py
```

---

## 15. Conclusion

This project demonstrates a complete fake review detection pipeline including preprocessing, feature engineering, model training, model persistence, and deployment using Streamlit.
