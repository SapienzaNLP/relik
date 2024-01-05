![](https://drive.google.com/uc?export=view&id=1vmf1rhGvc5JQjV1EbCKP66G79NiJMs17)

<div align="center">    
 
# Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget

[![arXiv](https://img.shields.io/badge/arXiv-placeholder-b31b1b.svg)](https://arxiv.org/abs/placeholder)

[![Open in Visual Studio Code](https://img.shields.io/badge/preview%20in-vscode.dev-blue)](https://github.dev/SapienzaNLP/relik)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Upload to PyPi](https://github.com/SapienzaNLP/relik/actions/workflows/python-publish-pypi.yml/badge.svg)](https://github.com/SapienzaNLP/relik/actions/workflows/python-publish-pypi.yml)
[![PyPi Version](https://img.shields.io/github/v/release/SapienzaNLP/relik)](https://github.com/SapienzaNLP/relik/releases)

</div>

A blazing fast and lightweight Information Extraction model for Entity Linking and Relation Extraction.


## Installation

Installation from PyPI
```bash
pip install relik
```

<details>
  <summary>Other installation options</summary>

#### Install with optional dependencies

Install with all the optional dependencies.

```bash
pip install relik[all]
```

Install with optional dependencies for training and evaluation.

```bash
pip install relik[train]
```

Install with optional dependencies for [FAISS](https://github.com/facebookresearch/faiss)

```bash
pip install relik[faiss] # or relik[faiss-gpu] for GPU support
```

Install with optional dependencies for serving the models with [FastAPI](https://fastapi.tiangolo.com/) and [Ray](https://docs.ray.io/en/latest/serve/quickstart.html).

```bash
pip install relik[serve]
```

#### Installation from source

```bash
git clone https://github.com/SapienzaNLP/relik.git
cd relik
pip install -e .[all]
```

</details>

## Usage

### Training

#### Retriever

##### Data Preparation

- TODO

##### Training the model

- TODO

#### Inference

- TODO

#### Reader

##### Data Preparation

Add candidates to the dataset

```bash
python scripts/data/add_candidates.py \
  --encoder riccorl/retriever-relik-entity-linking-aida-wikipedia-base-question-encoder \
  --index riccorl/retriever-relik-entity-linking-aida-wikipedia-base-index \
  --input_path path/to/processed/data \
  --output_path ... \
  # --index_device cuda \
  # --precision 16 \
  # --topics \
  # --log_recall
```

##### Training the model

- TODO

#### Inference

- TODO

### Inference

[//]: # (Write a short description of the model and how to use it with the `from_pretrained` method.)

```python
from relik import Relik

relik = Relik.from_pretrained("path/to/relik/model")
relik("Michael Jordan was one of the best players in the NBA.")
```

Retrievers and Readers can be used separately:

```python
from relik import Relik

# If you want to use the retriever only
retriever = Relik.from_pretrained("path/to/relik/model", reader=None)

# If you want to use the reader only
reader = Relik.from_pretrained("path/to/relik/model", retriever=None)
```

or

```python
from relik.retriever import GoldenRetriever

retriever = GoldenRetriever(
    question_encoder="path/to/relik/retriever-question-encoder",
    document_index="path/to/relik/retriever-document-index",
)
```
