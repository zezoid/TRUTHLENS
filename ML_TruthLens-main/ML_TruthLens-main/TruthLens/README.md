# TruthLens – Fake News Detection Using NLP and Machine Learning

TruthLens is a machine-learning-only academic project that classifies news articles as fake or real using text preprocessing, TF-IDF feature extraction, and multiple supervised classifiers.

## 1. Project overview
This project is designed for MCA students and focuses on a practical, explainable, reproducible NLP workflow. It covers dataset integration, exploratory data analysis, preprocessing, feature engineering, model comparison, evaluation, and prediction.

The system does not claim to verify factual truth; it learns statistical patterns from historical text data. It can help compare several models, identify the best-performing one, and make predictions on new article text.

## 2. Problem statement
Fake news spreads quickly and can influence public opinion. This project builds a text classification system that detects patterns in news content and predicts whether an article is likely fake or real.

## 3. Objectives
- Load and merge fake and real news datasets.
- Clean and normalize the text.
- Extract features using TF-IDF.
- Train and compare multiple classifiers.
- Evaluate models using standard metrics.
- Save the best model and deploy prediction logic.
- Document the full workflow for academic presentation.

## 4. Scope
- ML-only project using Python in Jupyter Notebook.
- CPU-based implementation suitable for macOS and Apple Silicon.
- No frontend, backend, or database system.

## 5. Technologies used
- Python 3
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn, SciPy
- NLTK
- Scikit-learn
- WordCloud
- Joblib

## 6. Dataset information
This project is built to work with the ISOT Fake and Real News Dataset.

Files expected:
- `datasets/raw/Fake.csv`
- `datasets/raw/True.csv`

If the files are not present, download them from Kaggle and place them in the `datasets/raw/` directory.

## 7. System architecture
The project is organized as follows:

- `datasets/raw/` – original source files
- `datasets/processed/` – processed dataset output
- `notebooks/` – analysis and training notebooks
- `src/` – reusable Python modules
- `models/` – saved trained model(s)
- `outputs/` – charts and evaluation reports

## 8. Installation and How to Run the Project

Follow these steps exactly to run the project on macOS/Linux with Python 3.

### Step 1: Open Terminal
```bash
cd /Users/rishi/Documents/C++/ML_Project/TruthLens
```

### Step 2: Create a virtual environment
```bash
python3 -m venv venv
```

### Step 3: Activate the virtual environment
```bash
source venv/bin/activate
```

### Step 4: Install project dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Download NLTK resources
```bash
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4 stopwords averaged_perceptron_tagger_eng
```

### Step 6: Download the dataset
This project uses the ISOT Fake and Real News Dataset.

Download the dataset from Kaggle and place the files here:
- `datasets/raw/Fake.csv`
- `datasets/raw/True.csv`

If the download is not available, use the Kaggle dataset link mentioned in the project documentation and save the files into the raw folder before training.

### Step 7: Start Jupyter Notebook
```bash
jupyter notebook
```

### Step 8: Run the notebooks in this order
Open the notebooks from the `notebooks/` folder and run them one by one in sequence:

1. `01_data_collection.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_exploratory_data_analysis.ipynb`
4. `04_feature_extraction.ipynb`
5. `05_classification_models.ipynb`
6. `06_kmeans_clustering.ipynb`
7. `07_model_evaluation.ipynb`
8. `08_model_prediction.ipynb`

### Step 9: Run reusable Python scripts (optional)
After the notebooks execute successfully, you can also use the source files directly:

```bash
python src/train.py
python src/predict.py
```

### Step 10: Check outputs
After running the notebooks, check these folders:
- `datasets/processed/` for the processed dataset
- `outputs/graphs/` for charts
- `outputs/confusion_matrices/` for confusion matrices
- `outputs/reports/` for evaluation CSV reports
- `models/` for the saved best model

### Expected result
When the project runs correctly, you should see:
- the processed dataset saved,
- graphs generated,
- classification metrics produced,
- the best model saved as `models/best_model.joblib`,
- predictions working on new article text.

### Common issues and solutions
- Problem: `Fake.csv` or `True.csv` not found
  - Solution: download the dataset and save both files inside `datasets/raw/`

- Problem: NLTK data missing
  - Solution: run the `python -m nltk.downloader ...` command again

- Problem: Jupyter not installed
  - Solution: run `pip install -r requirements.txt` after activating the virtual environment

- Problem: module not found error
  - Solution: ensure the virtual environment is active and all dependencies are installed

- Problem: model not saved
  - Solution: run the training notebooks completely in order before prediction

## 9. Execution workflow
1. Place the raw CSV files in `datasets/raw/`.
2. Open the notebooks in order:
   - 01_data_collection.ipynb
   - 02_data_preprocessing.ipynb
   - 03_exploratory_data_analysis.ipynb
   - 04_feature_extraction.ipynb
   - 05_classification_models.ipynb
   - 06_kmeans_clustering.ipynb
   - 07_model_evaluation.ipynb
   - 08_model_prediction.ipynb
3. Run each notebook sequentially.
4. Use `src/train.py` or `src/predict.py` for reusable operations.

## 10. Model description
The system compares:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Multinomial Naive Bayes
- Linear SVM

The best model is selected using measured performance, especially macro F1-score and fake-class recall.

## 11. Evaluation methodology
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Cross-validation on training data

## 12. Limitations
- The model learns patterns from historical data; it does not prove truth.
- Cross-source generalization can be limited when training and testing distributions differ.
- Source-related patterns and repeated articles must be handled carefully.

## 13. Results
Results are generated only after executing the notebooks on the actual dataset. Do not fabricate metrics or charts.

## 14. Future enhancements
- Add topic-aware features.
- Compare title-only and article-body-only models.
- Evaluate transformer-based models.
- Add model explainability.

## 15. Conclusion
TruthLens is a practical ML project that demonstrates text preprocessing, TF-IDF, classification, model comparison, and real-world evaluation in a way suitable for an MCA academic submission.

## 16. Files created
- `src/preprocessing.py`
- `src/feature_extraction.py`
- `src/train.py`
- `src/evaluate.py`
- `src/predict.py`
- `notebooks/01_data_collection.ipynb` to `08_model_prediction.ipynb`
