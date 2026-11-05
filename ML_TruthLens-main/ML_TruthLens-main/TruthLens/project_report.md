# TruthLens Project Report

## 1. Introduction
The spread of fake news has become a significant problem in modern digital communication. In the age of social media and instant publication, misinformation can influence public opinion, financial decisions, political outcomes, and social trust. Traditional fact-checking methods are slow and often difficult to scale. This creates the need for automatic, data-driven methods that can analyze textual content and estimate whether a news article is likely fake or real.

TruthLens is an academic machine learning project designed to detect fake news using Natural Language Processing (NLP) and supervised learning. The project uses the ISOT Fake and Real News Dataset, preprocesses text data, extracts TF-IDF features, compares multiple classification algorithms, and saves the best-performing model for later prediction. The project is intentionally structured for an MCA college environment and is suitable for practical examination, viva voce, and final demonstration.

## 2. Problem Statement
The problem is to build a machine learning system that can classify a news article as either fake news (0) or real news (1). This task is not a factual verification problem, but a text classification problem based on learned patterns from historical labeled data. The model analyzes linguistic patterns, word frequency, sentiment clues, and article style to estimate whether a given article resembles fake news or real reporting.

The project must also explain that the model does not independently verify whether a claim is true or false. Instead, it learns statistical patterns from the dataset and uses those patterns to make predictions. This is an important academic point because it avoids overclaiming the model’s ability.

## 3. Objectives of the Project
The major objectives of this project are:

- To collect and combine fake and real news datasets.
- To perform exploratory data analysis on the dataset.
- To clean and preprocess textual data before modeling.
- To transform text into numerical representation using TF-IDF.
- To train and compare multiple machine learning models.
- To evaluate model performance using standard classification metrics.
- To select the best model based on measured evidence.
- To save the trained model and use it for new predictions.
- To document the work in a way suitable for academic submission and viva.

## 4. Scope of the Project
The project is intentionally limited to machine learning and NLP. It does not include a website, backend system, REST API, database, or front-end application. The work is implemented in Python and is designed to run locally in Jupyter Notebook on a CPU-based environment such as an Apple Silicon MacBook.

The scope includes:

- Data acquisition and integration.
- Preprocessing of raw text.
- EDA and visualization.
- TF-IDF representation.
- Comparison of five supervised classifiers.
- Unsupervised K-Means clustering experiment.
- Evaluation using confusion matrices and classification reports.
- Final model saving and prediction support.

## 5. Dataset Description
The project uses the ISOT (Information Security and Object Technology) Fake and Real News Dataset, which is a widely used benchmark for fake news detection research.

The dataset contains the following file types:

- `Fake.csv`
- `True.csv`

These files contain columns such as:

- `title`
- `text`
- `subject`
- `date`

A target label column is created as:

- Fake article = 0
- Real article = 1

The requirement is to load both datasets using pandas, assign labels, concatenate them into one dataset, examine missing values, detect duplicates, and perform exploratory analysis.

## 6. Research and Data Quality Considerations
A realistic fake news detection project must be aware of dataset biases and limitations. In the ISOT dataset, there may be repeated articles, publisher-specific wording patterns, and source-specific signatures that influence classification. For example, some fake articles may reflect repeated writing styles or domain-specific vocabulary. This can create a misleading impression that the model understands truth, when it is actually learning patterns in language and source variation.

Therefore, it is important to:

- Exclude source or publisher identifiers from the final text-only model.
- Avoid using `subject` and `date` as primary model features unless explicitly tested.
- Keep the final evaluation set isolated from model tuning.
- Investigate duplicates and near-duplicates before training.
- Document limitations in the model’s generalization ability across different sources.

This makes the project academically correct and helps avoid over-claiming the performance of the model.

## 7. System Architecture
The project is organized into modular components:

- `datasets/raw/`: stores the original raw dataset files.
- `datasets/processed/`: stores the merged and cleaned dataset.
- `notebooks/`: contains the step-by-step Jupyter notebooks.
- `src/`: contains reusable Python modules for preprocessing, feature extraction, training, evaluation, and prediction.
- `outputs/`: stores graphs, confusion matrices, and evaluation reports.
- `models/`: stores the trained best model.

The main architecture follows a standard ML workflow:

1. Load raw dataset.
2. Merge fake and real news samples.
3. Clean and normalize text.
4. Create TF-IDF features.
5. Train multiple models.
6. Evaluate and compare results.
7. Save the best model.
8. Predict on new text.

## 8. Data Preprocessing Design
Text preprocessing is a crucial part of the project because classifier performance depends heavily on data quality and consistent cleaning.

The implemented preprocessing pipeline includes:

- Handling missing values.
- Removing duplicate entries.
- Lowercasing all text.
- Removing HTML tags.
- Removing URLs and email addresses.
- Removing special characters.
- Normalizing whitespace.
- Tokenization.
- Stopword removal.
- Lemmatization.
- Preserving negation words such as `not`, `no`, and `never`.

This pipeline is applied consistently before model training. The same cleaning function is used for prediction to ensure no mismatch between training and inference.

## 9. TF-IDF Feature Extraction
After cleaning, the text is converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).

In simple terms, TF-IDF measures how important a word is in a document relative to the whole dataset:

- Term Frequency (TF): how often a word appears in an article.
- Inverse Document Frequency (IDF): how rare the word is across the dataset.

Words that appear frequently in a single article but are not common across many articles carry higher importance. This is useful for fake news classification because certain terms, phrases, and writing patterns can help distinguish class behavior.

The project uses a scikit-learn `TfidfVectorizer` with configurable parameters such as:

- `max_features`
- `ngram_range`
- `min_df`
- `max_df`
- `stop_words`

The vectorizer must be fit only on the training set. This is essential to prevent data leakage and ensures the test set remains unseen during feature engineering.

## 10. Machine Learning Algorithms Used
The project compares five supervised classification models:

### 10.1 Logistic Regression
Logistic Regression is a linear classifier used for binary classification tasks. It estimates the probability that an input belongs to a class. It is simple, fast, and interpretable, which makes it suitable for this project.

### 10.2 Decision Tree Classifier
A Decision Tree splits the dataset based on feature values. It learns a hierarchical decision structure. The advantage is interpretability; the disadvantage is that it may overfit on noisy text data.

### 10.3 K-Nearest Neighbors (KNN)
KNN classifies a sample based on the labels of its nearest neighbors. For text data, KNN can be computationally heavy with high-dimensional sparse features. Therefore, the project uses a reproducible and justified approach, such as dimensionality reduction or a controlled sample size, where needed.

### 10.4 Multinomial Naive Bayes
Naive Bayes is widely used for text classification because it works well with word counts and frequency-based features. It is fast and often performs strongly on NLP tasks with sparse data.

### 10.5 Linear Support Vector Machine (LinearSVC)
Linear SVM finds the optimal hyperplane separating classes in high-dimensional space. It performs very well on sparse text features and is considered one of the strongest baselines for fake news detection.

## 11. K-Means Clustering Experiment
In addition to supervised models, the project includes an unsupervised K-Means clustering experiment. This is because clustering helps examine whether fake and real articles naturally separate in the feature space.

The steps include:

- Build TF-IDF features.
- Apply TruncatedSVD if needed for dimensionality reduction.
- Use K-Means with two clusters.
- Visualize cluster assignments.
- Compare the cluster labels with the real labeled classes.

It is important to understand that K-Means does not inherently know which cluster represents fake news or real news. The cluster names are assigned by interpretation after observing their composition.

## 12. Evaluation Metrics
The project evaluates all classification models using the standard measures listed below:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- ROC-AUC where supported
- Cross-validation on training data

Macro F1-score and fake-class recall are emphasized because both are useful when the classes may be unevenly distributed. In fake news detection, the ability to correctly detect fake articles is especially important.

## 13. Model Comparison Strategy
The final model is not selected by assumption. It is selected based on measured results. The project compares each model's performance using a consistent train-test split and evaluates them fairly.

The comparison table includes:

- Model name
- Accuracy
- Precision
- Recall
- F1-score
- Fake-class recall
- Macro F1-score
- ROC-AUC (if available)

The best-performing model is saved using `joblib` so that it can be reused without retraining.

## 14. Model Saving and Prediction
The project creates a reusable prediction pipeline that includes:

- Text preprocessing
- TF-IDF transformation
- Trained classifier

This avoids inconsistent transformation when making predictions on new text. The system can load the saved model and classify new article text as either:

- `FAKE NEWS`
- `REAL NEWS`

It is important to explain to the user that the model does not verify factual truth. It only assigns a class based on patterns learned from prior text data.

## 15. Project Workflow
The project is developed in a structured sequence:

### Phase 1: Project Setup and Dataset Loading
This stage includes setting up the workspace, creating project folders, and preparing the dataset inputs.

### Phase 2: Data Preprocessing and EDA
The project cleans text and explores features such as article length, label distribution, and word frequency.

### Phase 3: NLP and TF-IDF Feature Extraction
The cleaned text is transformed into numerical features for model training.

### Phase 4: Classification Algorithms
Five algorithms are trained and compared.

### Phase 5: K-Means Clustering
An unsupervised experiment is conducted to understand cluster behavior.

### Phase 6: Model Evaluation and Comparison
Standard metrics and visualizations are produced.

### Phase 7: Model Saving and Loading
The best pipeline is saved to disk and reloaded for testing.

### Phase 8: Documentation and Validation
The documentation is written and the implementation is tested for correctness.

## 16. Implementation Details
The project is implemented using Python and the following libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Scikit-learn
- NLTK
- WordCloud
- Joblib
- Jupyter Notebook

The project also follows quality practices such as:

- Reproducible random seeds.
- Stratified train-test splits.
- No manual data leakage.
- Modular and reusable code design.
- Clear file organization.
- Save outputs to the `outputs/` directory.

## 17. Execution and Validation
The project is designed to run locally using Python 3 and Jupyter Notebook. The workflow is:

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Download NLTK resources.
4. Place the raw CSV files in `datasets/raw/`.
5. Run notebooks in order.
6. Verify that graphs and reports are saved.
7. Confirm the saved model loads correctly.

The project was validated for Python syntax and notebook JSON structure. A smoke test of the workflow was also performed with a small synthetic dataset because the actual ISOT files were not available in the current execution environment. This confirms the pipeline can train, evaluate, and save a model, though the real Kaggle dataset must still be used to reproduce academic results.

## 18. Limitations of the System
This project has several limitations that are important to mention academically:

- It detects patterns in text, not objective truth.
- It may learn source-specific patterns rather than true fake content patterns.
- It may not generalize to unseen domains or new styles of misinformation.
- It is limited by the quality and balance of the training data.
- High test accuracy is not proof of universal fake news detection.

These limitations must be clearly explained during viva and project presentation.

## 19. Future Enhancements
Several improvements can be made in future versions of the project:

- Compare title-only and body-only models.
- Add cross-source validation experiments.
- Use transformer-based models such as BERT.
- Add explainability using SHAP or LIME.
- Create a larger robust benchmark across multiple datasets.
- Add real-time classification for live text input.

## 20. Syllabus Mapping

| Experiment | Topic | Coverage in TruthLens |
|---|---|---|
| Experiment 1 | NumPy, Pandas, Matplotlib, Seaborn, Scipy, data preprocessing | Yes |
| Experiment 2 | Linear Regression discussion | Not primary; not used for binary fake news classification |
| Experiment 3 | Logistic Regression, Decision Tree, KNN, confusion matrix, precision, recall, F1 | Yes |
| Experiment 4 | Naive Bayes, SVM, classification evaluation | Yes |
| Experiment 5 | K-Means clustering | Yes |
| Experiment 6 | NLP, tokenization, TF-IDF, text classification, model saving/loading | Yes |

## 21. Conclusion
TruthLens demonstrates a complete and academically suitable fake news detection pipeline in Python. It covers dataset preparation, text preprocessing, exploratory analysis, TF-IDF feature extraction, multiple machine learning classifiers, K-Means clustering, model evaluation, and prediction using a saved model. The project is appropriate for an MCA undergraduate practical and demonstrates core concepts of machine learning, NLP, data analysis, and model evaluation.

The project is valuable not only for technical implementation but also for understanding the role of data quality, bias awareness, statistical reasoning, and responsible AI usage. The implementation is intentionally transparent, modular, and reproducible, making it suitable for both academic assessment and practical demonstration.

## 22. Final Remarks
This project should be used as a strong academic foundation for a fake news detection system. It is not a factual verification engine; it is a pattern-learning classifier that analyzes textual content and predicts whether a news article resembles fake or real reporting based on historical data. This distinction is essential for honest and correct project reporting.
