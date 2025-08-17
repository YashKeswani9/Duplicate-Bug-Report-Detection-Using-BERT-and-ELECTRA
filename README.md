# Duplicate Bug Report Detection — NLP Pipeline (Data Prep + Model Training)

End‑to‑end notebooks for preparing text data and training transformer‑based models for duplicate bug report detection. The workflow covers classic NLP preprocessing, topic modeling & feature selection, class imbalance handling, and fine‑tuning BERT/ELECTRA with evaluation.

## 📂 Repository Structure

```
.
├── nlp_project_data_preparation.ipynb   # Data cleaning, preprocessing, topic modeling, feature selection, ADASYN oversampling
└── nlp_project_model_training.ipynb     # BERT/ELECTRA training (default & tuned), evaluation and reporting
```

## 🧠 What’s Inside

### 1) `nlp_project_data_preparation.ipynb`
- **Setup & Preparation** (Colab‑friendly; includes Google Drive mounting)
- **Preliminary Preprocessing**: lowercasing, tokenization, stopword removal, stemming/normalization, repeated‑token cleanup
- **Topic Modeling & Feature Selection**: LDA/Gensim or sklearn LDA to cluster reports by topic; per‑topic feature selection for stronger signals
- **Class Imbalance**: `imblearn` **ADASYN** oversampling to balance positive/negative classes
- **Outputs**: curated DataFrames ready for model training/evaluation

**Key libraries**: `nltk`, `gensim`, `scikit-learn`, `imblearn`, `scipy`, `pandas`, `numpy`

### 2) `nlp_project_model_training.ipynb`
- **Setup & Preparation** (Colab‑friendly; includes Google Drive mounting)
- **Models**:
  - **BERT** — baseline (default parameters)
  - **BERT (hyperparameter‑tuned)**
  - **ELECTRA (fine‑tuned)**
- **Evaluation**: accuracy, precision, recall, F1; dataset splits & metrics reporting
- **Artifacts**: trained model weights/checkpoints (if you save them during runs)

**Key libraries**: `transformers`, `torch`, `pandas`, `scikit-learn`

## 🚀 Getting Started

> You can run these notebooks either in **Google Colab** or **locally**. Colab is the quickest way to reproduce.

### Option A — Google Colab (recommended)
1. Upload this repository to your Google Drive or open the notebooks directly in Colab.
2. Run the **Setup** cell to mount Drive if prompted.
3. Set the input/output paths in the notebook cells (as needed) and run all cells **top to bottom** in each notebook, in order:
   1. `nlp_project_data_preparation.ipynb`
   2. `nlp_project_model_training.ipynb`

### Option B — Local environment
1. **Python 3.10+** is recommended.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -U pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or CUDA wheel if using GPU
   pip install transformers scikit-learn pandas numpy scipy nltk gensim imbalanced-learn matplotlib tqdm
   ```
4. Launch Jupyter:
   ```bash
   pip install jupyter
   jupyter notebook
   ```
5. Open and run the notebooks in the same order as above.

## 🗂️ Data

- Place your raw text/CSV files in a known path (e.g., under `data/` or your Drive path).
- Update any path variables at the top of the notebooks to point to your datasets.
- The **data preparation** notebook produces cleaned feature tables suitable for training (saved via pandas). The **training** notebook expects those outputs.

> **Note**: If you’re using an imbalanced dataset for duplicate vs non‑duplicate labels, keep ADASYN enabled in the data prep phase.

## ⚙️ Configuration Tips

- **Tokenizer & Max Length**: Ensure the tokenizer (BERT/ELECTRA) and `max_length` match across training and evaluation.
- **Batch Size**: Adjust per your GPU/CPU memory. Start small (e.g., 16) and scale up.
- **Learning Rate & Epochs**: Reasonable starting points are 2e‑5 to 5e‑5 for 3–5 epochs for BERT‑like models.
- **Reproducibility**: Set random seeds in numpy/torch and log your training/eval parameters.

## 📊 Metrics & Reporting

The training notebook computes **accuracy**, **precision**, **recall**, and **F1**. Consider persisting a classification report and confusion matrix for quick comparisons across runs.

## 📈 Results (from prior runs)

> These results are **reported from completed experiments** and included here for reference. The shared notebooks are docs‑only and don’t re‑run training by default.

| Dataset | Best Model | Accuracy |
|---|---|---|
| Android | BERT (tuned) | **90.20%** |
| Eclipse | BERT (fine‑tuned) | **83.92%** |

*Note:* Exact best‑model per dataset may vary by seed and tuning; see your experiment logs/checkpoints if you choose to include them later.

## 🧪 Extending the Work

- Try **per‑topic classifiers** (one model per LDA topic) vs a single global model.
- Add **feature unions**: classic TF‑IDF features side‑by‑side with transformer embeddings.
- Experiment with **threshold tuning** and **calibration** (e.g., Platt scaling).
- Perform **K‑fold** cross‑validation for more robust estimates.

## 💾 Saving Artifacts

If you save model checkpoints, push them via Git LFS or store externally (e.g., Drive, S3, or a model registry) to keep the repo lightweight.

## ✅ How to Reproduce (Quick Checklist)

- [ ] Run `nlp_project_data_preparation.ipynb` and export the processed training/validation/test frames
- [ ] Run `nlp_project_model_training.ipynb` to fine‑tune **BERT** and **ELECTRA**
- [ ] Capture metrics (accuracy/precision/recall/F1) and optionally persist a CSV/JSON log
- [ ] (Optional) Save best checkpoints & tokenizer to a `/models` folder or remote storage
