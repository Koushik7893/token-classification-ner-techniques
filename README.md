# 🏷️ Named Entity Recognition (Token Classification)

This project covers both theoretical concepts and hands-on implementation of **Named Entity Recognition (NER)** using various tagging schemes, decoding methods, embeddings, and a custom BiLSTM-CRF model trained on the CoNLL-2003 dataset.

---

## 📌 Highlights

* 🧾 **Tagging Schemes**: BIO, IOBES, BILOU
* 🔍 **Decoding Methods**: Softmax, CRF, Span-based
* 🔡 **Embeddings**: Static (GloVe), Contextual (ELMo, BERT), Character-level (CharCNN, Char-LSTM)
* 🧠 **Model**: BiLSTM + CRF
* 📊 **Evaluation**: F1 Score, Precision, Recall

---

## ⚙️ About the BiLSTM-CRF Model

The **BiLSTM-CRF** is a hybrid sequence labeling model that combines:

* **Word embeddings**: Capture semantic meaning of tokens.
* **Character embeddings + Char-BiLSTM**: Handle out-of-vocabulary words and learn subword features.
* **BiLSTM layer**: Captures contextual relationships in both directions.
* **CRF layer**: Applies structured prediction for label dependencies (e.g., "I-LOC" cannot follow "B-PER").

This architecture enhances performance by integrating both fine-grained and coarse-grained token information.

---

## 📁 Notebooks

| Notebook                             | Description                                           |
| ------------------------------------ | ----------------------------------------------------- |
| `01_tagging_schemes.ipynb`           | Explains BIO, IOBES, BILOU tagging formats            |
| `02_decoding_methods.ipynb`          | Demonstrates Softmax, CRF, and Span-based decoding    |
| `03_embedding_representations.ipynb` | Overview of embedding types (GloVe, ELMo, BERT, Char) |
| `bilstm_crf_conll2003.ipynb`      | End-to-end implementation of BiLSTM-CRF on CoNLL-2003 |

---

## 📂 Project Structure

```bash
token-classification-ner/
│
├── README.md
├── requirements.txt
├── notebooks/
│   ├── theory/
│   │   ├── 01_tagging_schemes.ipynb
│   │   ├── 02_decoding_methods.ipynb
│   │   ├── 03_embedding_representations.ipynb
│   └── bilstm_crf_conll2003.ipynb
```

---

## 📦 Dataset

The project uses the **CoNLL-2003** dataset. You can either load it via Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("conll2003")
```

Or download manually using:

```python
import kagglehub
path = kagglehub.dataset_download("juliangarratt/conll2003-dataset")
```

---

## 📒 Detailed Notebook Summaries

### ✅ `01_tagging_schemes.ipynb`

* Explanation of BIO, IOB, IOBES, and BILOU formats
* Tagging illustrations with token-label pairs
* Use case: Consistent labeling for NER sequences

---

### ✅ `02_decoding_methods.ipynb`

* Softmax: Assigns most probable label independently
* CRF: Considers label transitions for global optimal decoding
* Span-based: Extracts contiguous entity spans
* Includes comparison of decoding styles

---

### ✅ `03_embedding_representations.ipynb`

* Static Embeddings: GloVe (fixed, context-independent)
* Contextual Embeddings: ELMo, BERT (dynamic, context-aware)
* Character-level: CharCNN, Char-LSTM (subword-based)

---

### ✅ `bilstm_crf_conll2003.ipynb`

* Load & preprocess CoNLL-2003 dataset
* Vocabulary construction for word, character, and labels
* Dataset & Dataloader creation using PyTorch
* BiLSTM-CRF model definition and training loop
* Evaluation using F1 Score and predictions on sample sentences

---

## 📈 Example Output

```
F1 Score: 89.7%
Entities Extracted: PER, LOC, ORG, MISC
```

---

## 🔧 Installation

```bash
git clone https://github.com/Koushik7893/token-classification-ner.git
cd token-classification-ner
pip install -r requirements.txt
```

---

## 👤 Author

**Koushik Reddy**
🔗 [Hugging Face](https://huggingface.co/Koushim)
🔗 [LinkedIn](https://www.linkedin.com/in/koushik-reddy-k-790938257)

---

## 📜 License

This project is licensed under the [Apache License](LICENSE).

