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

## Inference

[//]: # (Write a short description of the model and how to use it with the `from_pretrained` method.)

ReLiK is a lightweight and fast model for **Entity Linking** and **Relation Extraction**. It is composed of two main components: a retriever and a reader. The retriever is responsible for retrieving relevant documents from a large collection of documents, while the reader is responsible for extracting entities and relations from the retrieved documents. ReLiK can be used with the `from_pretrained` method to load a pre-trained pipeline.

Here is an example of how to use ReLiK for Entity Linking:

```python
from relik import Relik

relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large")
relik("Michael Jordan was one of the best players in the NBA.")
```

and for Relation Extraction:

```python
from relik import Relik

relik = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-large")
relik("Michael Jordan was one of the best players in the NBA.")
```

The full list of available models can be found on [🤗 Hugging Face](https://huggingface.co/collections/sapienzanlp/relik-retrieve-read-and-link-665d9e4a5c3ecba98c1bef19).

Retrievers and Readers can be used separately:

```python
from relik import Relik

# If you want to use the retriever only
retriever = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-large", reader=None)

# If you want to use the reader only
reader = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-large", retriever=None)
```
<!-- 
or

```python
from relik.retriever import GoldenRetriever

# If you want to use the retriever only
retriever = GoldenRetriever(
    question_encoder="path/to/relik/retriever-question-encoder",
    document_index="path/to/relik/retriever-document-index",
)

from relik.reader import R
# If you want to use the reader only
reader = 
``` -->

## Training

Here we provide instructions on how to train the retriever and the reader.
All your data should have the following starting structure:

```jsonl
{
  "doc_id": int,  # Unique identifier for the document
  "doc_text": txt,  # Text of the document
  "doc_annotations": # Char level annotations
    [
      [start, end, label],
      [start, end, label],
      ...
    ]
}
```

### Retriever

We perform a two-step training process for the retriever. First, we "pre-train" the retriever using BLINK (Wu et al., 2019) dataset and then we "fine-tune" it using AIDA (Hoffart et al, 2011).

#### Data Preparation

The retriever requires a dataset in a format similar to [DPR](https://github.com/facebookresearch/DPR): a `jsonl` file where each line is a dictionary with the following keys:

```json lines
{
  "question": "....",
  "positive_ctxs": [{
    "title": "...",
    "text": "...."
  }],
  "negative_ctxs": [{
    "title": "...",
    "text": "...."
  }],
  "hard_negative_ctxs": [{
    "title": "...",
    "text": "...."
  }]
}
```

##### BLINK

The BLINK dataset can be downloaded from the [GENRE](https://github.com/facebookresearch/GENRE) repo from [here](https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh).
We used `blink-train-kilt.jsonl` and `blink-dev-kilt.jsonl` as training and validation datasets.
To convert the BLINK dataset to the ReLiK format, you can follow these steps:

1. Assuming the data is downloaded in `data/blink`, you can convert it to the ReLiK format with the following command:

```bash
# Train
python scripts/data/blink/preprocess_genre_blink.py \
  data/blink/blink-train-kilt.jsonl \
  data/blink/processed/blink-train-kilt-relik.jsonl

# Dev
python scripts/data/blink/preprocess_genre_blink.py \
  data/blink/blink-dev-kilt.jsonl \
  data/blink/processed/blink-dev-kilt-relik.jsonl
```

2. Then you can windowize the data with the following command:

```bash
# train
python scripts/data/create_windows.py \
  data/blink/processed/blink-train-kilt-relik.jsonl \
  data/blink/processed/blink-train-kilt-relik-windowed.jsonl

# dev
python scripts/data/create_windows.py \
  data/blink/processed/blink-dev-kilt-relik.jsonl \
  data/blink/processed/blink-dev-kilt-relik-windowed.jsonl
```

3. Finally, you can convert the data to the DPR format with the following command:

```bash
# train
python scripts/data/blink/convert_to_dpr.py \
  data/blink/processed/blink-train-kilt-relik-windowed.jsonl \
  data/blink/processed/blink-train-kilt-relik-windowed-dpr.jsonl

# dev
python scripts/data/blink/convert_to_dpr.py \
  data/blink/processed/blink-dev-kilt-relik-windowed.jsonl \
  data/blink/processed/blink-dev-kilt-relik-windowed-dpr.jsonl
```

##### AIDA

Since the AIDA dataset is not publicly available, we can provide the annotations for the AIDA dataset in the ReLiK format as an example.
Assuming you have the full AIDA dataset in the `data/aida`, you can convert it to the ReLiK format and then create the windows with the following script:

```bash
python scripts/data/create_windows.py \
  data/data/processed/aida-train-relik.jsonl \
  data/data/processed/aida-train-relik-windowed.jsonl
```

and then convert it to the DPR format:

```bash
python scripts/data/convert_to_dpr.py \
  data/data/processed/aida-train-relik-windowed.jsonl \
  data/data/processed/aida-train-relik-windowed-dpr.jsonl
```

#### Training the model

To train the model you need a configuration file. 
<!-- You can find an example in `relik/retriever/conf/finetune_iterable_in_batch.yaml`. -->
The configuration files in `relik/retriever/conf` are `pretrain_iterable_in_batch.yaml` and `finetune_iterable_in_batch.yaml`, which we used to pre-train and fine-tune the retriever, respectively.

To train the retriever on the AIDA dataset, you can run the following command:

```bash
relik retriever train relik/retriever/conf/finetune_iterable_in_batch.yaml \
  model.language_model=intfloat/e5-base-v2 \
  train_dataset_path=data/aida/processed/aida-train-relik-windowed-dpr.jsonl \
  val_dataset_path=data/aida/processed/aida-dev-relik-windowed-dpr.jsonl \
  test_dataset_path=data/aida/processed/aida-dev-relik-windowed-dpr.jsonl
```

The `relik retriever train` command takes the following arguments:

- `config_path`: The path to the configuration file.
- `overrides`: A list of overrides to the configuration file, in the format `key=value`.

### Inference

- TODO

### Reader

#### Data Preparation

The reader requires the windowized dataset we created in section [Retriever](#retriever) augmented with the candidate from the retriever. 

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

#### Training the model

Similar to the retriever, the reader requires a configuration file. The folder `relik/reader/conf` contains the configuration files we used to train the reader.
For instance, `large.yaml` is the configuration file we used to train the large reader.
By running the following command, you can train the reader on the AIDA dataset:

```bash
relik reader train relik/reader/conf/large.yaml \
  train_dataset_path=data/aida/processed/aida-train-relik-windowed-candidates.jsonl \
  val_dataset_path=data/aida/processed/aida-dev-relik-windowed-candidates.jsonl \
  test_dataset_path=data/aida/processed/aida-dev-relik-windowed-candidates.jsonl
```

### Inference

- TODO

## Performance

### Entity Linking

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
   - Select "Ma - strong annotation match"
   - In Name filed write the name you want to give to the experiment
   - In URI field write: [http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm](http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm)
   - Select the datasets (We use AIDA-B, MSNBC, Der, K50, R128, R500, OKE15, OKE16)
   - Finally, run experiment

### Relation Extraction

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
