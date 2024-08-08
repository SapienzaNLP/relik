<div align="center">
  <img src="https://github.com/SapienzaNLP/relik/blob/main/relik.png?raw=true" height="250">
  <img src="https://github.com/SapienzaNLP/relik/blob/main/Sapienza_Babelscape.png?raw=true" height="100">
</div>

<div align="center">

# Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget

[![Conference](http://img.shields.io/badge/ACL-2024-4b44ce.svg)](https://2024.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2408.00103-b31b1b.svg)](https://arxiv.org/abs/2408.00103)

[![relik](https://img.shields.io/badge/ReLiK-white?logo=data:image/png;base64,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)](https://github.com/SapienzaNLP/relik)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D)](https://huggingface.co/collections/sapienzanlp/relik-retrieve-read-and-link-665d9e4a5c3ecba98c1bef19)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FCD21D)](https://huggingface.co/spaces/relik-ie/Information-Extraction)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NnE_4zXV05I1zwGH0tSe7blnHsBiTy_2?usp=sharing)

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![PyPi Version](https://img.shields.io/badge/PyPI-3776AB?logo=pypi&logoColor=white)]([https://github.com/SapienzaNLP/relik/releases](https://pypi.org/project/relik/))
[![Release Version](https://img.shields.io/github/v/release/SapienzaNLP/relik)](https://github.com/SapienzaNLP/relik/releases)

</div>

A blazing fast and lightweight Information Extraction model for **Entity Linking** and **Relation Extraction**.

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

FAISS PyPI package is only available for CPU. For GPU, install it from source or use the conda package.

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


## 🤖 Models

New models:

- **ReLiK Small for Entity Linking (🆕🤏⚡ Tiny and Fast)**: [`sapienzanlp/relik-entity-linking-small`](https://huggingface.co/sapienzanlp/relik-entity-linking-small)
- **ReLiK Large for Closed Information Extraction (🔥 EL + RE)**: [`relik-ie/relik-cie-small`](https://huggingface.co/relik-ie/relik-cie-small)
- **ReLiK Large for Entity Linking (🔥 EL for the wild)**: [`relik-ie/relik-entity-linking-large-robust`](https://huggingface.co/relik-ie/relik-entity-linking-large-robust)
- **ReLiK Large for Entity Linking (🔥 RE + NER)**: [`relik-ie/relik-relation-extraction-small-wikipedia-ner`](https://huggingface.co/relik-ie/relik-relation-extraction-small-wikipedia-ner)

Models from the paper:

- **ReLiK Large for Entity Linking (📝 Paper version)**: [`sapienzanlp/relik-entity-linking-large`](https://huggingface.co/sapienzanlp/relik-entity-linking-large)
- **ReLik Base for Entity Linking (📝 Paper version)**: [`sapienzanlp/relik-entity-linking-base`](https://huggingface.co/sapienzanlp/relik-entity-linking-base)
- **ReLiK Large for Relation Extraction (📝 Paper version)**: [`sapienzanlp/relik-relation-extraction-nyt-large`](https://huggingface.co/sapienzanlp/relik-relation-extraction-nyt-large)

A full list of models can be found on [🤗 Hugging Face](https://huggingface.co/collections/sapienzanlp/relik-retrieve-read-and-link-665d9e4a5c3ecba98c1bef19).

Other models sizes will be available in the future 👀.


## 🚀 Quick Start

[//]: # (Write a short description of the model and how to use it with the `from_pretrained` method.)

ReLiK is a lightweight and fast model for **Entity Linking** and **Relation Extraction**.
It is composed of two main components: a retriever and a reader.
The retriever is responsible for retrieving relevant documents from a large collection,
while the reader is responsible for extracting entities and relations from the retrieved documents.
ReLiK can be used with the `from_pretrained` method to load a pre-trained pipeline.

Here is an example of how to use ReLiK for Entity Linking:

```python
from relik import Relik
from relik.inference.data.objects import RelikOutput

relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large")
relik_out: RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")
```

Output:

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

relik = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-nyt-large")
relik_out: RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")
```

Output:

    RelikOutput(
      text='Michael Jordan was one of the best players in the NBA.', 
      tokens=Michael Jordan was one of the best players in the NBA., 
      id=0, 
      spans=[
        Span(start=0, end=14, label='--NME--', text='Michael Jordan'), 
        Span(start=50, end=53, label='--NME--', text='NBA')
      ], 
      triplets=[
        Triplets(
          subject=Span(start=0, end=14, label='--NME--', text='Michael Jordan'), 
          label='company', 
          object=Span(start=50, end=53, label='--NME--', text='NBA'), 
          confidence=1.0
          )
      ], 
      candidates=Candidates(
        span=[], 
        triplet=[
                  [
                    [
                      {"text": "company", "id": 4, "metadata": {"definition": "company of this person"}}, 
                      {"text": "nationality", "id": 10, "metadata": {"definition": "nationality of this person or entity"}}, 
                      {"text": "child", "id": 17, "metadata": {"definition": "child of this person"}}, 
                      {"text": "founded by", "id": 0, "metadata": {"definition": "founder or co-founder of this organization, religion or place"}}, 
                      {"text": "residence", "id": 18, "metadata": {"definition": "place where this person has lived"}},
                      ...
                  ]
              ]
          ]
      ),
    )

### Usage

Retrievers and Readers can be used separately.
In the case of retriever-only ReLiK, the output will contain the candidates for the input text.

Retriever-only example:

```python
from relik import Relik
from relik.inference.data.objects import RelikOutput

# If you want to use only the retriever
retriever = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large", reader=None)
relik_out: RelikOutput = retriever("Michael Jordan was one of the best players in the NBA.")
```

Output:

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

Reader-only example:

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

Output:

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

ReLiK provides a CLI to serve a [FastAPI](https://fastapi.tiangolo.com/) server for the model or to perform inference on a dataset.

#### `relik serve`

```bash
relik serve --help

Usage: relik serve [OPTIONS] RELIK_PRETRAINED [DEVICE] [RETRIEVER_DEVICE]                             
                    [DOCUMENT_INDEX_DEVICE] [READER_DEVICE] [PRECISION]                                
                    [RETRIEVER_PRECISION] [DOCUMENT_INDEX_PRECISION]                                   
                    [READER_PRECISION] [ANNOTATION_TYPE]                                               
                                                                                                       
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────╮
│ *    relik_pretrained              TEXT                        [default: None] [required]           │
│      device                        [DEVICE]                    The device to use for relik (e.g.,   │
│                                                                'cuda', 'cpu').                      │
│                                                                [default: None]                      │
│      retriever_device              [RETRIEVER_DEVICE]          The device to use for the retriever  │
│                                                                (e.g., 'cuda', 'cpu').               │
│                                                                [default: None]                      │
│      document_index_device         [DOCUMENT_INDEX_DEVICE]     The device to use for the index      │
│                                                                (e.g., 'cuda', 'cpu').               │
│                                                                [default: None]                      │
│      reader_device                 [READER_DEVICE]             The device to use for the reader     │
│                                                                (e.g., 'cuda', 'cpu').               │
│                                                                [default: None]                      │
│      precision                     [PRECISION]                 The precision to use for relik       │
│                                                                (e.g., '32', '16').                  │
│                                                                [default: 32]                        │
│      retriever_precision           [RETRIEVER_PRECISION]       The precision to use for the         │
│                                                                retriever (e.g., '32', '16').        │
│                                                                [default: None]                      │
│      document_index_precision      [DOCUMENT_INDEX_PRECISION]  The precision to use for the index   │
│                                                                (e.g., '32', '16').                  │
│                                                                [default: None]                      │
│      reader_precision              [READER_PRECISION]          The precision to use for the reader  │
│                                                                (e.g., '32', '16').                  │
│                                                                [default: None]                      │
│      annotation_type               [ANNOTATION_TYPE]           The type of annotation to use (e.g., │
│                                                                'CHAR', 'WORD').                     │
│                                                                [default: char]                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────╮
│ --host                         TEXT     [default: 0.0.0.0]                                          │
│ --port                         INTEGER  [default: 8000]                                             │
│ --frontend    --no-frontend             [default: no-frontend]                                      │
│ --help                                  Show this message and exit.                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

For example:

```bash
relik serve sapienzanlp/relik-entity-linking-large
```

#### `relik inference`

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

### Docker Images

Docker images for ReLiK are available on [Docker Hub](https://hub.docker.com/r/sapienzanlp/relik). You can pull the latest image with:

```bash
docker pull sapienzanlp/relik:latest
```

and run the image with:

```bash
docker run -p 12345:8000 sapienzanlp/relik:latest -c relik-ie/relik-cie-small
```

The API will be available at `http://localhost:12345`. It exposes a single endpoint `/relik` with several parameters that can be passed to the model.
A quick documentation of the API can be found at `http://localhost:12345/docs`. Here is a simple example of how to query the API:

```bash
curl -X 'GET' \
  'http://127.0.0.1:12345/api/relik?text=Michael%20Jordan%20was%20one%20of%20the%20best%20players%20in%20the%20NBA.&is_split_into_words=false&retriever_batch_size=32&reader_batch_size=32&return_windows=false&use_doc_topic=false&annotation_type=char&relation_threshold=0.5' \
  -H 'accept: application/json'
```

Here the full list of parameters that can be passed to the docker image:

```bash
docker run sapienzanlp/relik:latest -h

Usage: relik [-h --help] [-c --config] [-p --precision] [-d --device] [--retriever] [--retriever-device] 
[--retriever-precision] [--index-device] [--index-precision] [--reader] [--reader-device] [--reader-precision] 
[--annotation-type] [--frontend] [--workers] -- start the FastAPI server for the RElik model

where:
    -h --help               Show this help text
    -c --config             Pretrained ReLiK config name (from HuggingFace) or path
    -p --precision          Precision, default '32'.
    -d --device             Device to use, default 'cpu'.
    --retriever             Override retriever model name.
    --retriever-device      Override retriever device.
    --retriever-precision   Override retriever precision.
    --index-device          Override index device.
    --index-precision       Override index precision.
    --reader                Override reader model name.
    --reader-device         Override reader device.
    --reader-precision      Override reader precision.
    --annotation-type       Annotation type ('char', 'word'), default 'char'.
    --frontend              Whether to start the frontend server.
    --workers               Number of workers to use.
```

## 📚 Before You Start

In the following sections, we provide a step-by-step guide on how to prepare the data, train the retriever and reader, and evaluate the model.

### Entity Linking

All your data should have the following structure:

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

The AIDA dataset is not publicly available, but we provide the file we used without `text` field. You can find the file in ReLiK format in `data/aida/processed` folder.

The Wikipedia index we used can be downloaded from [here](https://huggingface.co/sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-index/blob/main/documents.jsonl).

### Relation Extraction

All your data should have the following structure:

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

For Relation Extraction, we provide an example of how to preprocess the NYT dataset from [raw_nyt](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view) taken from [CopyRE](https://github.com/xiangrongzeng/copy_re?tab=readme-ov-file). Download the dataset to data/raw_nyt and then run:

```bash
python scripts/data/nyt/preprocess_nyt.py data/raw_nyt data/nyt/processed/
```

Please be aware that for fair comparison we reproduced the preprocessing from previous work, which leads to duplicate triplets due to the incorrect handling of repeated surface forms for entity spans. If you want to correctly parse the original data to ReLiK format, you can set the flag --legacy-format False. Just be aware that the provided RE NYT models were trained on the legacy format.

## 🦮 Retriever

We perform a two-step training process for the retriever. First, we "pre-train" the retriever using BLINK (Wu et al., 2019) dataset, and then we "fine-tune" it using AIDA (Hoffart et al, 2011).

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

The retriever also needs an index to search for the documents. The documents to index can be either a JSONL file or a TSV file similar to
[DPR](https://github.com/facebookresearch/DPR):

- `jsonl`: each line is a JSON object with the following keys: `id`, `text`, `metadata`
- `tsv`: each line is a tab-separated string with the `id` and `text` columns,
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

```bash
# train
relik data create-windows \
  data/blink/processed/blink-train-kilt-relik.jsonl \
  data/blink/processed/blink-train-kilt-relik-windowed.jsonl

# dev
relik data create-windows \
  data/blink/processed/blink-dev-kilt-relik.jsonl \
  data/blink/processed/blink-dev-kilt-relik-windowed.jsonl
```

and then convert it to the DPR format:

```bash
# train
relik data convert-to-dpr \
  data/blink/processed/blink-train-kilt-relik-windowed.jsonl \
  data/blink/processed/blink-train-kilt-relik-windowed-dpr.jsonl \
  data/kb/wikipedia/documents.jsonl \
  --title-map data/kb/wikipedia/title_map.json

# dev
relik data convert-to-dpr \
  data/blink/processed/blink-dev-kilt-relik-windowed.jsonl \
  data/blink/processed/blink-dev-kilt-relik-windowed-dpr.jsonl \
  data/kb/wikipedia/documents.jsonl \
  --title-map data/kb/wikipedia/title_map.json
```

##### AIDA

Since the AIDA dataset is not publicly available, we can provide the annotations for the AIDA dataset in the ReLiK format as an example.
Assuming you have the full AIDA dataset in the `data/aida`, you can convert it to the ReLiK format and then create the windows with the following script:

```bash
relik data create-windows \
  data/aida/processed/aida-train-relik.jsonl \
  data/aida/processed/aida-train-relik-windowed.jsonl
```

and then convert it to the DPR format:

```bash
relik data convert-to-dpr \
  data/aida/processed/aida-train-relik-windowed.jsonl \
  data/aida/processed/aida-train-relik-windowed-dpr.jsonl \
  data/kb/wikipedia/documents.jsonl \
  --title-map data/kb/wikipedia/title_map.json
```

#### Relation Extraction

##### NYT

```bash
relik data create-windows \
  data/data/processed/nyt/train.jsonl \
  data/data/processed/nyt/train-windowed.jsonl \
  --is-split-into-words \
  --window-size none 
```

and then convert it to the DPR format:

```bash
relik data convert-to-dpr \
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

```bash
relik retriever train relik/retriever/conf/finetune_iterable_in_batch.yaml \
  model.language_model=intfloat/e5-base-v2 \
  data.train_dataset_path=data/aida/processed/aida-train-relik-windowed-dpr.jsonl \
  data.val_dataset_path=data/aida/processed/aida-dev-relik-windowed-dpr.jsonl \
  data.test_dataset_path=data/aida/processed/aida-test-relik-windowed-dpr.jsonl \
  data.shared_params.documents_path=data/kb/wikipedia/documents.jsonl
```

#### Relation Extraction

The configuration file in `relik/retriever/conf` is `finetune_nyt_iterable_in_batch.yaml`, which we used to fine-tune the retriever for the NYT dataset. For cIE we repurpose the one pretrained from BLINK in the previous step.

For instance, to train the retriever on the NYT dataset, you can run the following command:

```bash
relik retriever train relik/retriever/conf/finetune_nyt_iterable_in_batch.yaml \
  model.language_model=intfloat/e5-base-v2 \
  data.train_dataset_path=data/nyt/processed/nyt-train-relik-windowed-dpr.jsonl \
  data.val_dataset_path=data/nyt/processed/nyt-dev-relik-windowed-dpr.jsonl \
  data.test_dataset_path=data/nyt/processed/nyt-test-relik-windowed-dpr.jsonl
```

### Inference

By passing `train.only_test=True` to the `relik retriever train` command, you can skip the training and only evaluate the model.
It needs also the path to the PyTorch Lightning checkpoint and the dataset to evaluate on.

```bash
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

With `push_to_hub=True` the model will be pushed to the 🤗 Hugging Face Hub with `repo_id` as the repository id where the model will be pushed.

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
The `RelikReaderForSpanExtraction` is used for span extraction, i.e. Entity Linking, while the `RelikReaderForTripletExtraction` is used for triplet extraction, i.e. Relation Extraction.

### Data Preparation

The reader requires the windowized dataset we created in Section [Before You Start](#before-you-start) augmented with the candidates from the retriever.
The candidates can be added to the dataset using the `relik retriever add-candidates` command.

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

```bash
relik retriever add-candidates sapienzanlp/relik-retriever-e5-base-v2-aida-blink-encoder sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-index data/aida/processed/aida-train-relik-windowed.jsonl data/aida/processed/aida-train-relik-windowed-candidates.jsonl
```

#### Relation Extraction

The same thing happens for Relation Extraction. If you want to use our trained retriever:

```bash
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

```bash
relik reader train relik/reader/conf/large.yaml \
  train_dataset_path=data/aida/processed/aida-train-relik-windowed-candidates.jsonl \
  val_dataset_path=data/aida/processed/aida-dev-relik-windowed-candidates.jsonl \
  test_dataset_path=data/aida/processed/aida-dev-relik-windowed-candidates.jsonl
```

#### Relation Extraction

The configuration files in `relik/reader/conf` are `large_nyt.yaml`, `base_nyt.yaml`, and `small_nyt.yaml`, which we used to train the large, base and small reader, respectively.
For instance, to train the large reader on the AIDA dataset run:

```bash
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

with `push_to_hub=True` the model will be pushed to the 🤗 Hugging Face Hub with `repo_id` as the repository id where the model will be uploaded.

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

| Model                                                                                 | AIDA     | MSNBC    | Der      | K50      | R128     | R500     | O15      | O16      | Tot      | OOD      | AIT (m:s) |
| ------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- |
| GENRE                                                                                 | 83.7     | 73.7     | 54.1     | 60.7     | 46.7     | 40.3     | 56.1     | 50.0     | 58.2     | 54.5     | 38:00     |
| EntQA                                                                                 | 85.8     | 72.1     | 52.9     | 64.5     | **54.1** | 41.9     | 61.1     | 51.3     | 60.5     | 56.4     | 20:00     |
| [ReLiK<sub>small<sub>](https://huggingface.co/sapienzanlp/relik-entity-linking-small) | 82.2     | 72.7     | 55.6     | 68.3     | 48.0     | 42.3     | 62.7     | 53.6     | 60.7     | 57.6     | 00:29     |
| [ReLiK<sub>Base<sub>](https://huggingface.co/sapienzanlp/relik-entity-linking-base)   | 85.3     | 72.3     | 55.6     | 68.0     | 48.1     | 41.6     | 62.5     | 52.3     | 60.7     | 57.2     | 00:29     |
| [ReLiK<sub>Large<sub>](https://huggingface.co/sapienzanlp/relik-entity-linking-large) | **86.4** | **75.0** | **56.3** | **72.8** | 51.7     | **43.0** | **65.1** | **57.2** | **63.4** | **60.2** | 01:46     |

Comparison systems' evaluation (InKB Micro F1) on the *in-domain* AIDA test set and *out-of-domain* MSNBC (MSN), Derczynski (Der), KORE50 (K50), N3-Reuters-128 (R128), 
N3-RSS-500 (R500), OKE-15 (O15), and OKE-16 (O16) test sets. **Bold** indicates the best model. 
GENRE uses mention dictionaries. 
The AIT column shows the time in minutes and seconds (m:s) that the systems need to process the whole AIDA test set using an NVIDIA RTX 4090, 
except for EntQA which does not fit in 24GB of RAM and for which an A100 is used.

To evaluate ReLiK we use the following steps:

1. Download the GERBIL server from [here](https://drive.google.com/file/d/1PvSlXke2cp_Jn-UgxIA8M9xN1G0Hv6ap/view?usp=sharing).

2. Start the GERBIL server:

```bash
cd gerbil && ./start.sh
```

2. Start the following services:

```bash
cd gerbil-SpotWrapNifWS4Test && mvn clean -Dmaven.tomcat.port=1235 tomcat:run
```

3. Start the ReLiK server for GERBIL providing the model name as an argument (e.g. `sapienzanlp/relik-entity-linking-large`):

```bash
python relik/reader/utils/gerbil.py --relik-model-name sapienzanlp/relik-entity-linking-large
```

4. Open the URL [http://localhost:1234/gerbil](http://localhost:1234/gerbil) and:
   - Select A2KB as experiment type
   - Select "Ma - strong annotation match"
   - In the Name field write the name you want to give to the experiment
   - In the URI field write: [http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm](http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm)
   - Select the datasets (We use AIDA-B, MSNBC, Der, K50, R128, R500, OKE15, OKE16)
   - Finally, run experiment

### Relation Extraction

The following table shows the results (Micro F1) of ReLiK Large on the NYT dataset:

| Model                                                                                          | NYT      | NYT (Pretr) | AIT (m:s) |
| ---------------------------------------------------------------------------------------------- | -------- | ----------- | --------- |
| REBEL                                                                                          | 93.1     | 93.4        | 01:45     |
| UiE                                                                                            | 93.5     | --          | --        |
| USM                                                                                            | 94.0     | 94.1        | --        |
| [ReLiK<sub>Large<sub>](https://huggingface.co/sapienzanlp/relik-relation-extraction-nyt-large) | **95.0** | **94.9**    | 00:30     |

To evaluate Relation Extraction we can directly use the reader with the script relik/reader/trainer/predict_re.py, pointing at the file with already retrieved candidates. If you want to use our trained Reader:

```bash
python relik/reader/trainer/predict_re.py --model_path sapienzanlp/relik-reader-deberta-v3-large-nyt --data_path /Users/perelluis/Documents/relik/data/debug/test.window.candidates.jsonl --is-eval
```

Be aware that we compute the threshold for predicting relations based on the development set. To compute it while evaluating you can run the following:

```bash
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

The data and software are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
