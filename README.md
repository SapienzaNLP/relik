![](https://drive.google.com/uc?export=view&id=1UwPIfBrG021siM9SBAku2JNqG4R6avs6)

<div align="center">

# Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget

[![Conference](http://img.shields.io/badge/ACL-2024-4b44ce.svg)](https://2024.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/)
[![arXiv](https://img.shields.io/badge/arXiv-placeholder-b31b1b.svg)](https://arxiv.org/abs/placeholder)

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D)](https://huggingface.co/collections/sapienzanlp/relik-retrieve-read-and-link-665d9e4a5c3ecba98c1bef19)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FCD21D)](https://huggingface.co/spaces/sapienzanlp/relik-demo)

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Upload to PyPi](https://github.com/SapienzaNLP/relik/actions/workflows/python-publish-pypi.yml/badge.svg)](https://github.com/SapienzaNLP/relik/actions/workflows/python-publish-pypi.yml)
[![PyPi Version](https://img.shields.io/github/v/release/SapienzaNLP/relik)](https://github.com/SapienzaNLP/relik/releases)

<a href="https://nlp.uniroma1.it/"><img src="https://img.shields.io/badge/Sapienza NLP-802433.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAMAAABrrFhUAAADAFBMVEUAAAD///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////8HPQsIAAAA/3RSTlMAAQIDBAUGBwgJCgsMDQ4PEBESExQVFhcYGRobHB0eHyAhIiMkJSYnKCkqKywtLi8wMTIzNDU2Nzg5Ojs8PT4/QEFCQ0RFRkdISUpLTE1O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"></a>
  <a href="https://babelscape.it/"><img src="https://img.shields.io/badge/Babelscape-215489.svg?logo=data:image/png;base64,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"></a>

</div>

A blazing fast and lightweight Information Extraction model for Entity Linking and Relation Extraction.

## 🛠️ Installation

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

FAISS pypi package is only available for CPU. If you want to use GPU, you need to install it from source or use the conda package.

For CPU:

```bash
pip install relik[faiss]
```

For GPU:

```bash
conda create -n relik python=3.10
conda activate relik

# install pytorch
conda install -y pytorch=2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# GPU
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
# or GPU with NVIDIA RAFT
conda install -y -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0

pip install relik
```

Install with optional dependencies for serving the models with
[FastAPI](https://fastapi.tiangolo.com/) and [Ray](https://docs.ray.io/en/latest/serve/quickstart.html).

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

## 🚀 Quick Start

[//]: # (Write a short description of the model and how to use it with the `from_pretrained` method.)

ReLiK is a lightweight and fast model for **Entity Linking** and **Relation Extraction**.
It is composed of two main components: a retriever and a reader.
The retriever is responsible for retrieving relevant documents from a large collection of documents,
while the reader is responsible for extracting entities and relations from the retrieved documents.
ReLiK can be used with the `from_pretrained` method to load a pre-trained pipeline.

Here is an example of how to use ReLiK for Entity Linking:

```python
from relik import Relik
from relik.inference.data.objects import RelikOutput

relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large")
relik_out: RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")
```

    RelikOutput(
      text="Michael Jordan was one of the best players in the NBA.",
      tokens=['Michael', 'Jordan', 'was', 'one', 'of', 'the', 'best', 'players', 'in', 'the', 'NBA', '.'],
      id=0,
      spans=[
          Span(start=0, end=14, label="Michael Jordan", text="Michael Jordan"),
          Span(start=50, end=53, label="National Basketball Association", text="NBA"),
      ],
      triples=[],
      candidates=Candidates(
          span=[
              [
                  [
                      {"text": "Michael Jordan", "id": 4484083},
                      {"text": "National Basketball Association", "id": 5209815},
                      {"text": "Walter Jordan", "id": 2340190},
                      {"text": "Jordan", "id": 3486773},
                      {"text": "50 Greatest Players in NBA History", "id": 1742909},
                      ...
                  ]
              ]
          ]
      ),
    )

and for Relation Extraction:

```python
from relik import Relik
from relik.inference.data.objects import RelikOutput

relik = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-large")
relik_out: RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")
```

### Models

Models can be found on [🤗 Hugging Face](https://huggingface.co/collections/sapienzanlp/relik-retrieve-read-and-link-665d9e4a5c3ecba98c1bef19).

- **ReLiK Large for Entity Linking**: [`sapienzanlp/relik-entity-linking-large`](https://huggingface.co/sapienzanlp/relik-entity-linking-large)
- **ReLik Base for Entity Linking**: [`sapienzanlp/relik-entity-linking-base`](https://huggingface.co/sapienzanlp/relik-entity-linking-base)
- **ReLiK Large for Relation Extraction**: [`sapienzanlp/relik-relation-extraction-large`](https://huggingface.co/sapienzanlp/relik-relation-extraction-large)

### Usage

Retrievers and Readers can be used separately.
In the case of retriever-only ReLiK, the output will contain the candidates for the input text.

```python
from relik import Relik
from relik.inference.data.objects import RelikOutput

# If you want to use only the retriever
retriever = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large", reader=None)
relik_out: RelikOutput = retriever("Michael Jordan was one of the best players in the NBA.")

```

    RelikOutput(
      text="Michael Jordan was one of the best players in the NBA.",
      tokens=['Michael', 'Jordan', 'was', 'one', 'of', 'the', 'best', 'players', 'in', 'the', 'NBA', '.'],
      id=0,
      spans=[],
      triples=[],
      candidates=Candidates(
          span=[
                  [
                      {"text": "Michael Jordan", "id": 4484083},
                      {"text": "National Basketball Association", "id": 5209815},
                      {"text": "Walter Jordan", "id": 2340190},
                      {"text": "Jordan", "id": 3486773},
                      {"text": "50 Greatest Players in NBA History", "id": 1742909},
                      ...
                  ]
          ],
          triplet=[],
      ),
    )

```python
from relik import Relik
from relik.inference.data.objects import RelikOutput

# If you want to use only the reader
reader = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large", retriever=None)
candidates = [
    "Michael Jordan",
    "National Basketball Association",
    "Walter Jordan",
    "Jordan",
    "50 Greatest Players in NBA History",
]
text = "Michael Jordan was one of the best players in the NBA."
relik_out: RelikOutput = reader(text, candidates=candidates)
```

    RelikOutput(
      text="Michael Jordan was one of the best players in the NBA.",
      tokens=['Michael', 'Jordan', 'was', 'one', 'of', 'the', 'best', 'players', 'in', 'the', 'NBA', '.'],
      id=0,
      spans=[
          Span(start=0, end=14, label="Michael Jordan", text="Michael Jordan"),
          Span(start=50, end=53, label="National Basketball Association", text="NBA"),
      ],
      triples=[],
      candidates=Candidates(
          span=[
              [
                  [
                      {
                          "text": "Michael Jordan",
                          "id": -731245042436891448,
                      },
                      {
                          "text": "National Basketball Association",
                          "id": 8135443493867772328,
                      },
                      {
                          "text": "Walter Jordan",
                          "id": -5873847607270755146,
                          "metadata": {},
                      },
                      {"text": "Jordan", "id": 6387058293887192208, "metadata": {}},
                      {
                          "text": "50 Greatest Players in NBA History",
                          "id": 2173802663468652889,
                      },
                  ]
              ]
          ],
      ),
    )

### CLI

ReLiK provides a CLI to perform inference on a text file or a directory of text files. The CLI can be used as follows:

```bash
relik inference --help

  Usage: relik inference [OPTIONS] MODEL_NAME_OR_PATH INPUT_PATH OUTPUT_PATH

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_name_or_path      TEXT  [default: None] [required]                                           │
│ *    input_path              TEXT  [default: None] [required]                                           │
│ *    output_path             TEXT  [default: None] [required]                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────╮
│ --batch-size                               INTEGER  [default: 8]                                        │
│ --num-workers                              INTEGER  [default: 4]                                        │
│ --device                                   TEXT     [default: cuda]                                     │
│ --precision                                TEXT     [default: fp16]                                     │
│ --top-k                                    INTEGER  [default: 100]                                      │
│ --window-size                              INTEGER  [default: None]                                     │
│ --window-stride                            INTEGER  [default: None]                                     │
│ --annotation-type                          TEXT     [default: char]                                     │
│ --progress-bar        --no-progress-bar             [default: progress-bar]                             │
│ --model-kwargs                             TEXT     [default: None]                                     │
│ --inference-kwargs                         TEXT     [default: None]                                     │
│ --help                                              Show this message and exit.                         │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

For example:

```bash
relik inference sapienzanlp/relik-entity-linking-large data.txt output.jsonl
```

## 📚 Before You Start

In the following sections, we provide a step-by-step guide on how to prepare the data, train the retriever and reader, and evaluate the model.

### Entity Linking

All your data should have the following starting structure:

```jsonl
{
  "doc_id": int,  # Unique identifier for the document
  "doc_text": txt,  # Text of the document
  "doc_span_annotations": # Char level annotations
    [
      [start, end, label],
      [start, end, label],
      ...
    ]
}
```

We used BLINK (Wu et al., 2019) and AIDA (Hoffart et al, 2011) datasets for training and evaluation.
More specifically, we used the BLINK dataset for pre-training the retriever and the AIDA dataset for fine-tuning the retriever and training the reader.

The BLINK dataset can be downloaded from the [GENRE](https://github.com/facebookresearch/GENRE) repo using this
[script](https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh).
We used `blink-train-kilt.jsonl` and `blink-dev-kilt.jsonl` as training and validation datasets.
Assuming we have downloaded the two files in the `data/blink` folder, we converted the BLINK dataset to the ReLiK format using the following script:

```console
# Train
python scripts/data/blink/preprocess_genre_blink.py \
  data/blink/blink-train-kilt.jsonl \
  data/blink/processed/blink-train-kilt-relik.jsonl

# Dev
python scripts/data/blink/preprocess_genre_blink.py \
  data/blink/blink-dev-kilt.jsonl \
  data/blink/processed/blink-dev-kilt-relik.jsonl
```

The AIDA dataset is not publicly available, but we provide the file we used without `text` field. You can find the file in ReLiK format in `data/aida/processed` folder.

The Wikipedia index we used can be downloaded from [here](https://huggingface.co/sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-index/blob/main/documents.jsonl).

### Relation Extraction

All your data should have the following starting structure:

```jsonl
{
  "doc_id": int,  # Unique identifier for the document
  "doc_words: list[txt] # Tokenized text of the document
  "doc_span_annotations": # Token level annotations of mentions (label is optional)
    [
      [start, end, label],
      [start, end, label],
      ...
    ],
  "doc_triplet_annotations": # Triplet annotations
  [
    {
      "subject": [start, end, label], # label is optional
      "relation": name, # type is optional
      "object": [start, end, label], # label is optional
    },
    {
      "subject": [start, end, label], # label is optional
      "relation": name, # type is optional
      "object": [start, end, label], # label is optional
    },
  ]
}
```

For Relation Extraction, we provide an example on how to preprocess the NYT datase from [raw_nyt](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view) taken from the [CopyRE](https://github.com/xiangrongzeng/copy_re?tab=readme-ov-file). Download the dataset to data/raw_nyt and then run:

```console
scripts/data/nyt/preprocess_nyt.py data/raw_nyt data/nyt/processed/
```

Please be aware that for fair comparison we reproduce the preprocessing from previous work, which leads to duplicate triplets due to the wrong handling of repeated surface forms for entity spans. If you want to correctly parse the original data to relik format you can set the flag --legacy-format False. Just be aware that the provided RE NYT models were trained on the legacy format.

## 🦮 Retriever

We perform a two-step training process for the retriever. First, we "pre-train" the retriever using BLINK (Wu et al., 2019) dataset and then we "fine-tune" it using AIDA (Hoffart et al, 2011).

### Data Preparation

The retriever requires a dataset in a format similar to [DPR](https://github.com/facebookresearch/DPR): a `jsonl` file where each line is a dictionary with the following keys:

```jsonl
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

The retriever also needs an index to search for the documents. The documents to index can be either a jsonl file or a tsv file similar to
[DPR](https://github.com/facebookresearch/DPR):

- `jsonl`: each line is a json object with the following keys: `id`, `text`, `metadata`
- `tsv`: each line is a tab-separated string with the `id` and `text` column,
  followed by any other column that will be stored in the `metadata` field

`jsonl` example:

```json lines
{
  "id": "...",
  "text": "...",
  "metadata": ["{...}"]
},
...
```

`tsv` example:

```tsv
id \t text \t any other column
...
```

#### Entity Linking

##### BLINK

Once you have the BLINK dataset in the ReLiK format, you can create the windows with the following script:

```console
# train
python scripts/data/create_windows.py \
  data/blink/processed/blink-train-kilt-relik.jsonl \
  data/blink/processed/blink-train-kilt-relik-windowed.jsonl

# dev
python scripts/data/create_windows.py \
  data/blink/processed/blink-dev-kilt-relik.jsonl \
  data/blink/processed/blink-dev-kilt-relik-windowed.jsonl
```

and then convert it to the DPR format:

```console
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

```console
python scripts/data/create_windows.py \
  data/data/processed/aida-train-relik.jsonl \
  data/data/processed/aida-train-relik-windowed.jsonl
```

and then convert it to the DPR format:

```console
python scripts/data/convert_to_dpr.py \
  data/data/processed/aida-train-relik-windowed.jsonl \
  data/data/processed/aida-train-relik-windowed-dpr.jsonl
```

#### Relation Extraction

##### NYT

```console
python scripts/data/create_windows.py \
  data/data/processed/nyt/train.jsonl \
  data/data/processed/nyt/train-windowed.jsonl \
  --is-split-into-words \
  --window-size none 
```

and then convert it to the DPR format:

```console
python scripts/data/convert_to_dpr.py \
  data/data/processed/nyt/train-windowed.jsonl \
  data/data/processed/nyt/train-windowed-dpr.jsonl
```

### Training the model

The `relik retriever train` command can be used to train the retriever. It requires the following arguments:

- `config_path`: The path to the configuration file.
- `overrides`: A list of overrides to the configuration file, in the format `key=value`.

Examples of configuration files can be found in the `relik/retriever/conf` folder.

#### Entity Linking

<!-- You can find an example in `relik/retriever/conf/finetune_iterable_in_batch.yaml`. -->
The configuration files in `relik/retriever/conf` are `pretrain_iterable_in_batch.yaml` and `finetune_iterable_in_batch.yaml`, which we used to pre-train and fine-tune the retriever, respectively.

For instance, to train the retriever on the AIDA dataset, you can run the following command:

```console
relik retriever train relik/retriever/conf/finetune_iterable_in_batch.yaml \
  model.language_model=intfloat/e5-base-v2 \
  train_dataset_path=data/aida/processed/aida-train-relik-windowed-dpr.jsonl \
  val_dataset_path=data/aida/processed/aida-dev-relik-windowed-dpr.jsonl \
  test_dataset_path=data/aida/processed/aida-test-relik-windowed-dpr.jsonl
```

#### Relation Extraction

The configuration files in `relik/retriever/conf` is `finetune_nyt_iterable_in_batch.yaml`, which we used to fine-tune the retriever for the NYT dataset. For cIE we repurpose the one pretrained from BLINK in the previous step.

For instance, to train the retriever on the NYT dataset, you can run the following command:

```console
relik retriever train relik/retriever/conf/finetune_nyt_iterable_in_batch.yaml \
  model.language_model=intfloat/e5-base-v2 \
  train_dataset_path=data/nyt/processed/nyt-train-relik-windowed-dpr.jsonl \
  val_dataset_path=data/nyt/processed/nyt-dev-relik-windowed-dpr.jsonl \
  test_dataset_path=data/nyt/processed/nyt-test-relik-windowed-dpr.jsonl
```


### Inference

By passing `train.only_test=True` to the `relik retriever train` command, you can skip the training and only evaluate the model.
It needs also the path to the PyTorch Lightning checkpoint and the dataset to evaluate on.

```console
relik retriever train relik/retriever/conf/finetune_iterable_in_batch.yaml \
  train.only_test=True \
  test_dataset_path=data/aida/processed/aida-test-relik-windowed-dpr.jsonl
  model.checkpoint_path=path/to/checkpoint
```

The retriever encoder can be saved from the checkpoint with the following command:

```python
from relik.retriever.lightning_modules.pl_modules import GoldenRetrieverPLModule

checkpoint_path = "path/to/checkpoint"
retriever_folder = "path/to/retriever"

# If you want to push the model to the Hugging Face Hub set push_to_hub=True
push_to_hub = False
# If you want to push the model to the Hugging Face Hub set the repo_id
repo_id = "sapienzanlp/relik-retriever-e5-base-v2-aida-blink-encoder"

pl_module = GoldenRetrieverPLModule.load_from_checkpoint(checkpoint_path)
pl_module.model.save_pretrained(retriever_folder, push_to_hub=push_to_hub, repo_id=repo_id)
```

with `push_to_hub=True` the model will be pushed to the 🤗 Hugging Face Hub with `repo_id` the repository id where the model will be pushed.

The retriever needs an index to search for the documents. The index can be created using `relik retriever build-index` command

```bash
relik retriever build-index --help 

 Usage: relik retriever build-index [OPTIONS] QUESTION_ENCODER_NAME_OR_PATH                                                                   
                                    DOCUMENT_PATH OUTPUT_FOLDER                                                                                                                                              
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    question_encoder_name_or_path      TEXT  [default: None] [required]                                                                   │
│ *    document_path                      TEXT  [default: None] [required]                                                                   │
│ *    output_folder                      TEXT  [default: None] [required]                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --document-file-type                                  TEXT     [default: jsonl]                                                            │
│ --passage-encoder-name-or-path                        TEXT     [default: None]                                                             │
│ --indexer-class                                       TEXT     [default: relik.retriever.indexers.inmemory.InMemoryDocumentIndex]          │
│ --batch-size                                          INTEGER  [default: 512]                                                              │
│ --num-workers                                         INTEGER  [default: 4]                                                                │
│ --passage-max-length                                  INTEGER  [default: 64]                                                               │
│ --device                                              TEXT     [default: cuda]                                                             │
│ --index-device                                        TEXT     [default: cpu]                                                              │
│ --precision                                           TEXT     [default: fp32]                                                             │
│ --push-to-hub                     --no-push-to-hub             [default: no-push-to-hub]                                                   │
│ --repo-id                                             TEXT     [default: None]                                                             │
│ --help                                                         Show this message and exit.                                                 │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

With the encoder and the index, the retriever can be loaded from a repo id or a local path:

```python
from relik.retriever import GoldenRetriever

encoder_name_or_path = "sapienzanlp/relik-retriever-e5-base-v2-aida-blink-encoder"
index_name_or_path = "sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-index"

retriever = GoldenRetriever(
  question_encoder=encoder_name_or_path,
  document_index=index_name_or_path,
  device="cuda", # or "cpu"
  precision="16", # or "32", "bf16"
  index_device="cuda", # or "cpu"
  index_precision="16", # or "32", "bf16"
)
```

and then it can be used to retrieve documents:

```python
retriever.retrieve("Michael Jordan was one of the best players in the NBA.", top_k=100)
```

## 🤓 Reader

The reader is responsible for extracting entities and relations from documents from a set of candidates (e.g., possible entities or relations).
The reader can be trained for span extraction or triplet extraction.
The `RelikReaderForSpanExtraction` is used for span extraction, i.e. Entity Linking , while the `RelikReaderForTripletExtraction` is used for triplet extraction, i.e. Relation Extraction.

### Data Preparation

The reader requires the windowized dataset we created in section [Before You Start](#before-you-start) augmented with the candidate from the retriever.
The candidate can be added to the dataset using the `relik retriever add-candidates` command.

```bash
relik retriever add-candidates --help

 Usage: relik retriever add-candidates [OPTIONS] QUESTION_ENCODER_NAME_OR_PATH                                 
                                       DOCUMENT_NAME_OR_PATH INPUT_PATH                                        
                                       OUTPUT_PATH

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    question_encoder_name_or_path      TEXT  [default: None] [required]                                    │
│ *    document_name_or_path              TEXT  [default: None] [required]                                    │
│ *    input_path                         TEXT  [default: None] [required]                                    │
│ *    output_path                        TEXT  [default: None] [required]                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --passage-encoder-name-or-path                           TEXT     [default: None]                           │
│ --relations                                              BOOLEAN  [default: False]                          │
│ --top-k                                                  INTEGER  [default: 100]                            │
│ --batch-size                                             INTEGER  [default: 128]                            │
│ --num-workers                                            INTEGER  [default: 4]                              │
│ --device                                                 TEXT     [default: cuda]                           │
│ --index-device                                           TEXT     [default: cpu]                            │
│ --precision                                              TEXT     [default: fp32]                           │
│ --use-doc-topics                  --no-use-doc-topics             [default: no-use-doc-topics]              │
│ --help                                                            Show this message and exit.               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### Entity Linking

We need to add candidates to each window that will be used by the Reader, using our previously trained Retriever. Here is an example using our already trained retriever on Aida for the train split:

```console
relik retriever add-candidates sapienzanlp/relik-retriever-e5-base-v2-aida-blink-encoder sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-index data/aida/processed/aida-train-relik-windowed.jsonl data/aida/processed/aida-train-relik-windowed-candidates.jsonl
```

#### Relation Extraction

The same thing happens for Relation Extraction. If you want to use our already trained retriever:

```console
relik retriever add-candidates sapienzanlp/relik-retriever-small-nyt-question-encoder sapienzanlp/relik-retriever-small-nyt-document-index data/nyt/processed/nyt-train-relik-windowed.jsonl data/nyt/processed/nyt-train-relik-windowed-candidates.jsonl
```

### Training the model

Similar to the retriever, the `relik reader train` command can be used to train the retriever. It requires the following arguments:

- `config_path`: The path to the configuration file.
- `overrides`: A list of overrides to the configuration file, in the format `key=value`.

Examples of configuration files can be found in the `relik/reader/conf` folder.

#### Entity Linking

The configuration files in `relik/reader/conf` are `large.yaml` and `base.yaml`, which we used to train the large and base reader, respectively.
For instance, to train the large reader on the AIDA dataset run:

```console
relik reader train relik/reader/conf/large.yaml \
  train_dataset_path=data/aida/processed/aida-train-relik-windowed-candidates.jsonl \
  val_dataset_path=data/aida/processed/aida-dev-relik-windowed-candidates.jsonl \
  test_dataset_path=data/aida/processed/aida-dev-relik-windowed-candidates.jsonl
```

#### Relation Extraction

The configuration files in `relik/reader/conf` are `large_nyt.yaml`, `base_nyt.yaml` and `small_nyt.yaml`, which we used to train the large, base and small reader, respectively.
For instance, to train the large reader on the AIDA dataset run:

```console
relik reader train relik/reader/conf/large_nyt.yaml \
  train_dataset_path=data/nyt/processed/nyt-train-relik-windowed-candidates.jsonl \
  val_dataset_path=data/nyt/processed/nyt-dev-relik-windowed-candidates.jsonl \
  test_dataset_path=data/nyt/processed/nyt-test-relik-windowed-candidates.jsonl
```

### Inference

The reader can be saved from the checkpoint with the following command:

```python
from relik.reader.lightning_modules.relik_reader_pl_module import RelikReaderPLModule

checkpoint_path = "path/to/checkpoint"
reader_folder = "path/to/reader"

# If you want to push the model to the Hugging Face Hub set push_to_hub=True
push_to_hub = False
# If you want to push the model to the Hugging Face Hub set the repo_id
repo_id = "sapienzanlp/relik-reader-deberta-v3-large-aida"

pl_model = RelikReaderPLModule.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)
pl_model.relik_reader_core_model.save_pretrained(experiment_path, push_to_hub=push_to_hub, repo_id=repo_id)
```

with `push_to_hub=True` the model will be pushed to the 🤗 Hugging Face Hub with `repo_id` the repository id where the model will be pushed.

The reader can be loaded from a repo id or a local path:

```python
from relik.reader import RelikReaderForSpanExtraction, RelikReaderForTripletExtraction

# the reader for span extraction
reader_span = RelikReaderForSpanExtraction(
  "sapienzanlp/relik-reader-deberta-v3-large-aida"
)
# the reader for triplet extraction
reader_tripltes = RelikReaderForTripletExtraction(
  "sapienzanlp/relik-reader-deberta-v3-large-nyt"
)
```

and used to extract entities and relations:

```python
# an example of candidates for the reader
candidates = ["Michael Jordan", "NBA", "Chicago Bulls", "Basketball", "United States"]
reader_span.read("Michael Jordan was one of the best players in the NBA.", candidates=candidates)
```

## 📊 Performance

### Entity Linking

We evaluate the performance of ReLiK on Entity Linking using [GERBIL](http://gerbil-qa.aksw.org/gerbil/). The following table shows the results (InKB Micro F1) of ReLiK Large and Base:

| Model | AIDA-B | MSNBC | Der   | K50   | R128  | R500  | OKE15 | OKE16 | AVG   | AVG-OOD | Speed (ms) |
| ----- | ------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------- | ---------- |
| Base  | 85.25  | 72.27 | 55.59 | 68.02 | 48.13 | 41.61 | 62.53 | 52.25 | 60.71 | 57.2    | n          |
| Large | 86.37  | 75.04 | 56.25 | 72.8  | 51.67 | 42.95 | 65.12 | 57.21 | 63.43 | 60.15   | n          |

To evaluate ReLiK we use the following steps:

1. Download the GERBIL server from [here](LINK).

2. Start the GERBIL server:

```console
cd gerbil && ./start.sh
```

2. Start the following services:

```console
cd gerbil-SpotWrapNifWS4Test && mvn clean -Dmaven.tomcat.port=1235 tomcat:run
```

3. Start the ReLiK server for GERBIL providing the model name as an argument (e.g. `sapienzanlp/relik-entity-linking-large`):

```console
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

To evalute Relation Extraction we can directly use the reader with the script relik/reader/trainer/predict_re.py, pointing at the file with already retrieved candidates. If you want to use our already trained Reader:

```console
python relik/reader/trainer/predict_re.py --model_path sapienzanlp/relik-reader-deberta-v3-large-nyt --data_path /Users/perelluis/Documents/relik/data/debug/test.window.candidates.jsonl --is-eval
```

Be aware that we compute the threshold for predicting relations based on the development set. To compute it while evaluating you can run:

```console
python relik/reader/trainer/predict_re.py --model_path sapienzanlp/relik-reader-deberta-v3-large-nyt --data_path /Users/perelluis/Documents/relik/data/debug/dev.window.candidates.jsonl --is-eval --compute-threshold
```

## 💽 Cite this work

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

## 🪪 License

The data is licensed under [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
