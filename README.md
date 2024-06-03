![](https://drive.google.com/uc?export=view&id=1UwPIfBrG021siM9SBAku2JNqG4R6avs6)

<div align="center">    
 
# Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget

[![Conference](http://img.shields.io/badge/ACL-2024-4b44ce.svg)](https://2024.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/)
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

### Inference

[//]: # (Write a short description of the model and how to use it with the `from_pretrained` method.)

ReLiK is a lightweight and fast model for **Entity Linking** and **Relation Extraction**. It is composed of two main components: a retriever and a reader. The retriever is responsible for retrieving relevant documents from a large collection of documents, while the reader is responsible for extracting entities and relations from the retrieved documents. ReLiK can be used with the `from_pretrained` method to load a pre-trained pipeline.

Here is an example of how to use ReLiK for Entity Linking:

```python
from relik import Relik

relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large")
relik("Michael Jordan was one of the best players in the NBA.")
```

To use ReLiK for Relation Extraction, you can use the `relik-relation-extraction-large` model:

```python
from relik import Relik

relik = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-large")
relik("Michael Jordan was one of the best players in the NBA.")
```

The full list of available models can be found on [🤗 Hugging Face](https://huggingface.co/collections/sapienzanlp/relik-retrieve-read-and-link-665d9e4a5c3ecba98c1bef19).

<!-- Retrievers and Readers can be used separately:

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
``` -->

### Training

#### Retriever

##### Data Preparation

- TODO

##### Training the model

To train the model you need a configuration file. You can find an example in `relik/retriever/conf/finetune_iterable_in_batch.yaml`. Once you have your configuration file, you can train the model with the following command:

```bash
relik retriever train path/to/config.yaml
```

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

To train the model you need a configuration file. You can find an example in `relik/reader/conf/large.yaml`. Once you have your configuration file, you can train the model with the following command:

```bash
relik reader train path/to/config.yaml
```

#### Inference

- TODO

### Performance

#### Entity Linking

We evaluate the performance of ReLiK on Entity Linking using [GERBIL](http://gerbil-qa.aksw.org/gerbil/). The following table shows the results (InKB Micro F1) of ReLiK Large and Base:

| Model | AIDA-B | MSNBC | Der | K50 | R128 | R500 | OKE15 | OKE16 | AVG | AVG-OOD | Speed (ms) |
|-------|--------|-------|-----|-----|------|------|-------|-------|-----|---------|------------|
| Base | 85.87 | 71.85 | 55.5 | 67.2 | 49.23 | 41.54 | 62.57 | 53.93 | 60.96 | 57.4 | n |
| Large | 87.37 | 73.2 | 58.24 | 68.25 | 50.13 | 42.75 | 66.67 | 56.7 | 62.91 | 59.42 | n |

To evaluate ReLiK we use the following steps:

1. Start the GERBIL server:

```bash
cd gerbil && ./start.sh
```

2. Start the following services:

```bash
cd gerbil-SpotWrapNifWS4Test && mvn clean -Dmaven.tomcat.port=1235 tomcat:run
```

3. Start the ReLiK server for GERBIL providing the model name as an argument (e.g. `sapienzanlp/relik-entity-linking-large`):

```bash
python relik/reader/utils/gerbil_server.py --relik-model-name sapienzanlp/relik-entity-linking-large
```

4. Open the url [http://localhost:1234/gerbil](http://localhost:1234/gerbil) and:
   - Select A2KB as experiment type
   - Select "Ma - strong annotation match?
   - In Name filed write the name you want to give to the experiment
   - In URI field write: [http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm](http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm)
   - Select the datasets (We use AIDA-B, MSNBC, Der, K50, R128, R500, OKE15, OKE16)
   - Finally, run experiment

#### Relation Extraction

- TODO

## Cite this work

If you use any part of this work, please consider citing the paper as follows:

```bibtex
@inproceedings{orlando-etal-2024-relik,
    title     = "Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget",
    author    = "Orlando, Riccardo and Huguet Cabot, Pere-Llu{\'\i}s and Barba, Edoardo and Navigli, Roberto",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month     = aug,
    year      = "2024",
    address   = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
}
```

## License

TODO
<!-- The data is licensed under [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/). -->
