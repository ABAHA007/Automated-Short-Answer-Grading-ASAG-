# 🧠 MDocAgent: Multi-Modal Document Understanding (Customized)

A custom implementation of **MDocAgent**, a powerful Multi-Agent and Multi-Modal framework for document understanding tasks. This project includes extraction, retrieval, and reasoning over documents using text and images.

---

## 📌 Overview

**MDocAgent** is designed to:
- Extract and preprocess document data (PDFs, scientific papers, etc.)
- Retrieve relevant textual and visual evidence
- Infer answers through a collaborative multi-agent setup

> Built using Python, PyTorch, Hugging Face Transformers, and OpenAI's LLMs.

---

## 📁 Project Structure

```
.
├── data/                  # Downloaded datasets (from HuggingFace)
├── scripts/               # Extraction, retrieval, prediction, evaluation
├── config/                # YAML config files for model and dataset settings
├── results/               # Inference & evaluation outputs
├── install.sh             # Dependency installer (Linux/Mac)
├── requirements.txt       # Python packages (Windows use)
├── README.md              # Project documentation
```

---

## 🚀 Installation

### 🧱 Create a Conda environment (Windows/Mac/Linux)

```bash
conda create -n mdocagent python=3.12 -y
conda activate mdocagent
```

### 💻 Install dependencies

For Windows:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install opencv-python scikit-learn matplotlib pyyaml
```

Or use:
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

1. Create a `data/` folder:
   ```bash
   mkdir data
   cd data
   ```

2. Download a dataset (e.g., [PTab](https://huggingface.co/datasets/aiming/PTab)) from Hugging Face and place it in the `data/` directory.

3. Run extraction:
   ```bash
   python scripts/extract.py --config-name ptab
   ```

---

## 🔍 Retrieval (Text / Image)

Edit `config/base.yaml` to switch between:

```yaml
defaults:
  - retrieval: text  # or 'image'
```

Then run:

```bash
python scripts/retrieve.py --config-name ptab
```

---

## 🧠 Inference

```bash
python scripts/predict.py --config-name ptab run-name=custom-run
```

Optional (limit to top-4):
```bash
python scripts/predict.py --config-name ptab run-name=custom-run dataset.top_k=4
```

---

## 📊 Evaluation

Add your OpenAI API key to:
```
config/model/openai.yaml
```

Run evaluation:

```bash
python scripts/eval.py --config-name ptab run-name=custom-run
```

---

## 📈 Results

Evaluation scores will be saved in:
```
results/ptab/custom-run/results.txt
```

---

## 🧪 Sample Commands

```bash
python scripts/extract.py --config-name ptab
python scripts/retrieve.py --config-name ptab
python scripts/predict.py --config-name ptab run-name=myrun dataset.top_k=4
python scripts/eval.py --config-name ptab run-name=myrun
```

---

## 📜 Citation

If you use this project, please cite:

```bibtex
@article{han2025mdocagent,
  title={MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding},
  author={Han, Siwei and Xia, Peng and Zhang, Ruiyi and Sun, Tong and Li, Yun and Zhu, Hongtu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2503.13964},
  year={2025}
}
```

---

## 📌 Acknowledgements

- [Original MDocAgent Repository](https://github.com/aiming-lab/MDocAgent)
- Hugging Face Datasets
- OpenAI GPT API
- PyTorch & Transformers

---

## 📄 License

This project is licensed under the MIT License.
