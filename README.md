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

[//]: # (Write a short description of the model and how to use it with the `from_pretrained` method.)

### Pretrained Pipelines

#### Entity Linking

We have four pretrained models for entity linking available on HuggingFace:

| Pipeline                                            | Description                                                                                                        | # Parameters  | Download                                                                         |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|---------------|----------------------------------------------------------------------------------|
| `riccorl/relik-entity-linking-aida-wikipedia-tiny`  | Entity Linking on trained on AIDA and Wikipedia as KB, `e5-small-v2` as retriever and `deberta-v3-small` as reader | 33M + 141M    | [link](https://huggingface.co/riccorl/relik-entity-linking-aida-wikipedia-tiny)  |
| `riccorl/relik-entity-linking-aida-wikipedia-small` | Entity Linking on trained on AIDA and Wikipedia as KB, `e5-base-v2` as retriever and `deberta-v3-small` as reader  | 109M + 141M   | [link](https://huggingface.co/riccorl/relik-entity-linking-aida-wikipedia-small) |
| `riccorl/relik-entity-linking-aida-wikipedia-base`  | Entity Linking on trained on AIDA and Wikipedia as KB, `e5-base-v2` as retriever and `deberta-v3-base` as reader   | 109M + 183M   | [link](https://huggingface.co/riccorl/relik-entity-linking-aida-wikipedia-base)  |
| `riccorl/relik-entity-linking-aida-wikipedia-large` | Entity Linking on trained on AIDA and Wikipedia as KB, `e5-base-v2` as retriever and `deberta-v3-large` as reader  | 109M + 183M   | [link](https://huggingface.co/riccorl/relik-entity-linking-aida-wikipedia-large) |

```python
from relik import Relik

# Entity Linking
relik = Relik.from_pretrained("riccorl/relik-entity-linking-aida-wikipedia-tiny")
relik("Michael Jordan was one of the best players in the NBA.")
```

Retrievers and Readers can be used separately:

```python
from relik import Relik

# If you want to use the retriever only
retriever = Relik.from_pretrained("riccorl/relik-entity-linking-aida-wikipedia-tiny", reader=None)

# If you want to use the reader only
reader = Relik.from_pretrained("riccorl/relik-entity-linking-aida-wikipedia-tiny", retriever=None)
```

or

```python
from relik.retriever import GoldenRetriever

retriever = GoldenRetriever(
    question_encoder="riccorl/retriever-relik-entity-linking-aida-wikipedia-small-question-encoder",
    document_index="riccorl/retriever-relik-entity-linking-aida-wikipedia-small-index",
)
```

#### Relation Extraction

```python
from relik import Relik

# Relation Extraction
relik = Relik.from_pretrained("riccorl/relik-relation-extraction-nyt-small")
relik("Michael Jordan was one of the best players in the NBA.")
```

