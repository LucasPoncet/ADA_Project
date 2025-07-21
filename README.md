# ADA 2025 Project – arXiv Paper Category Classification

## Project Overview

This repository contains an assignment project for the *Advanced Data Analysis (ADA) 2025* course. The project implements a multiclass classification pipeline that categorizes scientific papers into high-level arXiv subject groups. Using metadata from arXiv, the system classifies each paper (based on its title and abstract) into one of eight broad arXiv categories. The goal is to explore different feature representations and models for this classification task, evaluate their performance, and discuss challenges like class imbalance.

## Dataset

The project uses the **arXiv Categories** dataset from Hugging Face, derived from arXiv metadata (paper titles, abstracts, and categories). Each paper in the dataset is labeled with one or more category tags (e.g., *cs.LG*, *astro-ph.CO*). For this project, only the **primary category** of each paper is used, so the task is simplified to single-label classification. The original dataset includes around 203,000 papers across 130 fine-grained categories (after filtering out very infrequent categories) and is already split into training, validation, and test sets (approximately 80/10/10 split). We further map each paper’s primary category to one of **8 high-level groups**, corresponding to top-level arXiv domains.

**High-Level Categories (8 classes)**: The mapping groups the detailed arXiv categories into eight broad domains. These include major fields like **Computer Science (cs)**, **Astrophysics (astro-ph)**, **Condensed Matter Physics (cond-mat)**, **Economics (econ)**, **Mathematics (math)**, **Other Physics (physics)** (covering remaining physics subfields like high-energy physics, etc.), **Quantitative Biology (q-bio)**, and **Quantitative Finance (q-fin)**. Each paper’s primary tag is mapped to one of these groups (e.g., a paper labeled *cs.LG* or *cs.AI* falls under **cs**, a paper in *astro-ph.GA* under **astro-ph**, and so on). This grouping helps focus on high-level classification rather than fine-grained categories.

## Preprocessing & Feature Engineering

For each paper, we construct text features from its **title and abstract**. Two different embedding strategies are implemented:

* **TF–IDF Vectorization:** We treat the paper’s title+abstract text as a document and apply TF–IDF (Term Frequency–Inverse Document Frequency) vectorization to obtain a sparse feature vector. We use unigrams and bigrams (ngram\_range=(1,2)) with a vocabulary limited to the top 50,000 features, ignoring very rare terms (min\_df=5) and using sublinear TF scaling. This produces high-dimensional sparse vectors representing the importance of words in the context of the corpus.

* **Transformer-Based Embeddings:** We use a pre-trained language model to generate a dense semantic embedding for the paper text. In particular, we experimented with the **Sentence-Transformers** library model *all-MiniLM-L6-v2* (a MiniLM model distilled for sentence embeddings) as well as **SciBERT** (a BERT variant with scientific vocabulary) to embed each title+abstract into a fixed-size vector. The implementation loads a Hugging Face transformer model and uses the CLS token’s output as the paper embedding. The maximum sequence length was set to 256 tokens, truncating longer abstracts for efficiency. These embeddings capture semantic information from the text in a compact dense vector (SciBERT produces a 768-dimensional vector for each paper).

**Category Label Encoding:** The paper’s primary category string is mapped to one of the 8 group labels and encoded as an integer (0–7) for modeling. For example, *"cs.LG"* → **cs** (label 2), *"astro-ph.CO"* → **astro-ph** (label 0), etc. This encoding is stored in the data as `category_number`. The processed dataset (after mapping) is saved in `data/processed/` as Parquet files for train, validation, and test splits, including the text and the encoded label.

## Models Implemented

We implemented and compared three different classification models in this pipeline:

* **Multilayer Perceptron (MLP):** A simple neural network classifier with one hidden layer. We feed the transformer-based embeddings into a fully-connected network (one hidden layer of size 512 with ReLU activation and dropout) and an output layer of size 8 (for the 8 categories). The network is trained with cross-entropy loss. We refer to this model as the *BERT MLP classifier* in code, since it operates on BERT/SciBERT embeddings. Training is done for up to 50 epochs with early stopping (patience=5) on the validation macro-F1 score.

* **Logistic Regression:** A linear classifier (with elastic-net regularization) trained on the TF–IDF features. We use scikit-learn’s LogisticRegression with a saga solver, tuning the regularization strength *C* (e.g., tried values like 2.0 or 4.0). This model handles the high-dimensional sparse input effectively. We monitor both accuracy and macro F1 on the validation set during training to pick the best model. (The best performing model is saved to `models/classifiers/logreg_elasticnet_tfidf.pkl`.)

* **XGBoost Classifier:** A gradient-boosted decision tree model (XGBoost) trained on the dense transformer embeddings. We configure XGBoost for multiclass (softmax) with 8 classes, using a modest tree depth (e.g. max\_depth=10) and 400 boosting rounds with early stopping. The model is trained with balanced instance weights to address class imbalance (described below). The trained booster is saved in JSON format (`models/classifiers/xgb_cls.json`).

## Evaluation

We evaluate model performance primarily using **Accuracy** and **Macro F1-score** on the validation and test sets. Accuracy measures overall correctness (the percentage of papers correctly classified), while Macro F1 gives the average F1-score across all categories treating each class equally important. Macro F1 is crucial here because of class imbalance: some categories (e.g., *Computer Science*) have far more examples than others (e.g., *Economics* or *Physics*:other). A high accuracy can be achieved by mostly getting the majority classes right, but a high macro-F1 indicates the model performs well across **all** classes.

During training, we observed that accuracy and macro-F1 can diverge due to imbalance. For example, the logistic regression might reach \~90% accuracy while the macro-F1 is much lower (because it struggles on the under-represented classes). We therefore tracked the **macro-F1** on the validation set to select models. The final results on the held-out test set include accuracy and macro-F1 for each model. As expected, the largest classes (cs, astro-ph, cond-mat) achieve higher per-class F1, whereas minor classes like econ or q-bio have lower recall and F1. We also report a full classification report (precision, recall, F1 for each class) and a confusion matrix for analysis. An example of the evaluation output for the logistic regression model is shown below (illustrating the class imbalance in support counts):

```
Class         Precision  Recall   F1-score  Support

    astro-ph      0.841     0.864     0.852      4129
    cond-mat      0.817     0.845     0.831      4223
          cs      0.960     0.924     0.941     11565
        econ      0.459     0.577     0.511       241
        math      0.389     0.538     0.452        91
     physics      0.312     0.556     0.400        45
       q-bio      0.224     0.311     0.260       103

    accuracy                          0.886     20397
   macro avg      0.572     0.659     0.607     20397
weighted avg      0.893     0.886     0.889     20397
...
Overall accuracy = 88.58 %   macro-F1 = 60.68%
```

**Class Imbalance:** The dataset is highly imbalanced – e.g., Computer Science makes up more than half of the examples, while categories like Mathematics, Physics (others), etc., appear in very small numbers. To mitigate this, we tried techniques such as class weighting. In the XGBoost model, we apply inverse class frequency weights so that errors on rare classes are penalized more during training. In logistic regression, scikit-learn’s `class_weight='balanced'` option or manual re-weighting could be used (though our final Logistic model used elastic-net regularization rather than explicit weighting). These measures help improve recall on minority classes, slightly boosting macro-F1.

## Visualization

To gain insight into the feature space, we visualized the paper embeddings using **t-SNE** (t-Distributed Stochastic Neighbor Embedding). We took the validation set’s SciBERT embeddings (768-dimensional) and projected them down to 2D with t-SNE for visualization. The resulting scatter plot (colored by true category) shows distinct clusters for some disciplines. For instance, **Computer Science** papers cluster tightly, indicating the model finds them linguistically similar, while **Astrophysics** vs **Condensed Matter** vs **Other Physics** form their own groupings. Smaller categories like **Economics** or **Quantitative Biology** might not form clear separate clusters due to their limited sample size, but we can still identify regions where they concentrate. This visualization (see figure below) provides a qualitative check that the embedding space is meaningfully structured: papers in the same high-level field tend to be closer together in the learned feature space.

&#x20;*t-SNE visualization of validation set embeddings. Each point represents a paper embedding (using SciBERT CLS vector), projected to 2D. Colors denote the 8 arXiv categories. We see well-separated clusters for some major fields (e.g., Computer Science in orange), while smaller categories are more diffuse.*

*(The figure above is saved as `experiments/figures/tsne_cls_validation.png` in the repo.)*

## Usage Instructions

To reproduce or extend this project, follow these steps:

1. **Setup Environment:** Ensure you have Python 3 (the code was tested with Python 3.11) and install the required libraries. You can install dependencies using the provided `requirements.txt` (it is a Conda environment export). For example:

   ```bash
   # Using conda
   conda create -n ada_project_env --file requirements.txt
   conda activate ada_project_env

   # (Alternatively, using pip for key packages)
   pip install -r requirements.txt
   ```

   Key dependencies include Hugging Face *datasets* (for data loading), *transformers* (for the SciBERT model), *sentence-transformers* (for MiniLM embeddings if used), *scikit-learn*, *XGBoost*, *PyTorch*, *Polars* (for data frame handling), and *Matplotlib/Seaborn* for plotting.

2. **Download/Prepare Dataset:** The arXiv data will be downloaded from HuggingFace on first use. Run the preprocessing script to fetch and prepare the data. For example:

   ```bash
   python src/data/data_step0.py
   ```

   This script (in `src/data/`) uses the HuggingFace `datasets.load_dataset("TimSchopf/arxiv_categories")` to download the dataset and then filters/maps the categories to the 8 groups. It will output processed data files under `data/processed/` (Parquet files for train, validation, test containing the text and label columns). Ensure you have an internet connection for the first run so the dataset can be fetched.

3. **Embed Text Data:** Next, generate the text embeddings for each split. You can use the `src/embed.py` script to create and save embeddings. By editing the hyperparameters at the top of `embed.py`, you can choose the method (`"tfidf"` or `"bert"`) and which split to process. For example, to generate SciBERT embeddings for the training set:

   ```bash
   python src/embed.py
   ```

   (Make sure `method` is set to `"bert"` and `split` to `"train_processed"` in the script.) This will produce a file like `data/embeddings/train_processed_bert.pkl`. Similarly run for `"validation_processed"` and `"test_processed"`. For TF-IDF vectors, set method to `"tfidf"`; on the first run it will fit a vectorizer and save it under `models/vectorizers/tfidf.pkl`. The script will save TF-IDF matrices (as `.pkl` via joblib) in `data/embeddings/` as well (e.g., `train_processed_tfidf.pkl`).

4. **Train Models:** With features in place, train the classifiers:

   * **Logistic Regression (TF-IDF):** Run `python src/model/main_TFIDF_LogReg.py`. This will load the TF-IDF training and validation matrices and train a logistic regression model, printing validation performance for configured hyperparameters. The best model is saved to `models/classifiers/logreg_elasticnet_tfidf.pkl`.
   * **XGBoost (BERT embeddings):** Run `python src/model/main_XGB_BERT.py`. This loads the BERT-based embeddings and trains an XGBoost model. Training progress and evaluation metrics (validation log-loss, etc.) will be displayed, and the final model is saved to `models/classifiers/xgb_cls.json`.
   * **MLP (BERT embeddings):** Run `python src/model/main_Bert.py`. This will train the neural network classifier on the BERT embeddings. It will output the best validation macro-F1 achieved and save the model’s weights to `models/classifiers/bert_mlp.pt`.
     You can adjust hyperparameters (like regularization strength, learning rate, number of epochs, etc.) by editing the respective scripts if needed.

5. **Evaluation and Outputs:** After training, you can evaluate the models on the test set. Use `python src/model/evaluate.py` to generate the metrics and confusion matrix. In that script, set `hp["method"]` to `"tfidf"`, `"xgb"`, or `"bert"` depending on which model you want to evaluate. Ensure the `emb_path`, `model_path`, and `parquet_path` in the script point to the test embeddings and the corresponding saved model. Running the script will print the accuracy and macro-F1 on the test set, output a detailed classification report (and save it as a CSV), and save a normalized confusion matrix plot in the `experiments/` folder. For example, if evaluating the XGBoost model, `evaluate.py` will output something like *“TEST accuracy = 0.90, macro-F1 = 0.54”* and save `experiments/eval_xgb_bert/classification_report.csv` and a confusion matrix image.

6. **Visualization:** To reproduce the t-SNE visualization, you can run `python src/t-sne_title_embedding.py`. This will load a sample of the validation set embeddings (by default) and produce a figure showing the 2D t-SNE plot colored by category. The plot image will be saved under `experiments/figures/` as `tsne_cls_validation.png`.

## Repository Structure

Below is an overview of the repository structure and key files:

* **`data/`** – Data storage directory (not included in git). After running preprocessing, it contains:

  * `processed/` – Processed dataset in Parquet format (`train_processed.parquet`, etc., each with text and category\_number).
  * `embeddings/` – Saved feature matrices (e.g., BERT embeddings `.pkl` and TF-IDF sparse matrices `.pkl` for each split).

* **`models/`** – Saved models and vectorizers:

  * `vectorizers/` – Contains the saved TF-IDF vectorizer (e.g., `tfidf.pkl`) after fitting on training data.
  * `classifiers/` – Contains saved model files like `logreg_elasticnet_tfidf.pkl`, `xgb_cls.json`, `bert_mlp.pt` after training.

* **`experiments/`** – Outputs from evaluation and analysis:

  * `eval_tfidf/`, `eval_xgb_bert/`, etc. – Subfolders with evaluation results (classification\_report.csv, confusion\_matrix.png for each model).
  * `figures/` – Plots and figures (e.g., the t-SNE plot image).

* **`src/`** – Source code for the project, organized by sub-module:

  * `src/data/` – Data loading and initial processing scripts. For example, `data_step0.py` downloads the dataset and creates the processed splits.
  * `src/preprocessing/` – (If present) might contain category mapping or filtering logic (e.g., mapping arXiv categories to the 8 groups).
  * `src/utils/` – Utility functions (e.g., `data_utils.py` for I/O, `embedding_utils.py` with TF-IDF and BERT encoding functions).
  * `src/model/` – Training and evaluation scripts for models:

    * `main_TFIDF_LogReg.py` – trains the Logistic Regression on TF-IDF features.
    * `main_XGB_BERT.py` – trains the XGBoost model on BERT embeddings.
    * `main_Bert.py` – trains the MLP classifier on BERT embeddings (uses `ClassesML/BertClassifier`).
    * `ClassesML/` – custom model classes (e.g., `BertClassifier` definition for the MLP, and a `TrainerClassifier` for training loop with early stopping).
    * `evaluate.py` – evaluation script to test a trained model and produce metrics/confusion matrix.
  * `t-sne_title_embedding.py` – script to generate t-SNE visualization of the embedding space.

* **`requirements.txt`** – List of required packages (Conda environment format). Use this to replicate the software environment.

*(Note: The actual data files are not tracked in the repo due to size; you will generate them by running the provided scripts. Likewise, the trained model files will appear in `models/` after you run training.)*

## License and Attribution

This is a course project, and no specific license is provided in the repository. The code is intended for educational purposes under the ADA 2025 course. If reusing this code, please attribute the original authors (course instructors and student).

**Data Attribution:** The arXiv metadata comes from the open arXiv dataset curated by Tim Schopf *et al.* and made available on Hugging Face. When using this data, please cite their work (see the dataset card on HuggingFace for citation details).

**Model and Tools:** This project leverages the Hugging Face Transformers library and pre-trained models. The SciBERT model is by AllenAI (“scibert\_scivocab\_uncased”), and *all-MiniLM-L6-v2* is a pre-trained MiniLM from the Sentence-Transformers library. We also used open-source libraries including scikit-learn, XGBoost, Polars (for data processing), and matplotlib/seaborn for plotting. We acknowledge these tools and their contributors.

## Optional Extensions and Future Work

While the current pipeline is functional, there are several opportunities to extend this project for improved performance or broader scope:

* **Incorporating Full Abstracts:** Our models currently use titles and abstracts (truncated to 256 tokens for BERT). Future work could explore using the **full** abstract text (longer input sequences or models that can handle long text) to see if that improves classification, especially for subtle distinctions between categories.

* **Multi-Label Classification:** The original task is inherently multi-label (a paper can belong to multiple categories). We simplified it to single-label by using the primary category. An extension would be to frame this as a **multi-label classification** problem, predicting *all* applicable categories for a paper. This would require adapting models (e.g., using sigmoid outputs and binary cross-entropy loss for each class) and evaluation metrics (e.g., subset accuracy or mean average precision).

* **Advanced Models & Transfer Learning:** We could try more advanced architectures or transfer learning approaches. For example, fine-tuning a transformer end-to-end on this classification task (instead of using frozen embeddings), or using a larger model like SciBERT-nli or PubMedBERT. Given the class imbalance, techniques like **data augmentation** or **semi-supervised learning** (e.g., using unlabeled arXiv papers or pseudo-labeling) could improve performance on low-resource classes. We could also experiment with **SetFit** or other few-shot learning methods as the FusionSent paper suggests, to better handle many classes with few examples.

* **Hyperparameter Tuning and Ensemble:** Further hyperparameter tuning (for XGBoost tree depth, regularization, etc.) might yield better results. Ensembling different models’ predictions could also be interesting to improve robustness (e.g., combining the logistic and XGBoost outputs).

* **Additional Visualizations:** We can explore other dimensionality reduction techniques (like PCA or UMAP) to visualize the embeddings, or even visualize the TF-IDF feature space. Analysis of the confusion matrix suggests which classes are most frequently confused – this could guide where the model needs improvement (for instance, if **q-bio** papers are often misclassified as **biology-related physics**, etc.).

By pursuing some of these extensions, one can further improve the classifier or adapt it to more realistic scenarios. This project provides a solid starting point for classifying scientific documents, and the above ideas could be explored as next steps in a research or production setting.
