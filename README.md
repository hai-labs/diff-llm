# DiffLLM

Extending the next token prediction setting into "diff prediction".

## Setup

Create virtual environment:

```
conda create -n diff-llm python=3.9
conda activate diff-llm
pip install -e .
```

Export secrets:

```
export $(grep -v '^#' secrets.txt | xargs)
```

Export env vars:


## Usage

### Create dataset

Debugging dataset

```
python diff_llm/create_dataset.py \
    --output-dir ./datasets/diff_corpus_small \
    --page-names '["Deep learning", "Ancient Greece", "Ted Chiang"]' \
    --n-revisions 1000
```

Small dataset for development
```
python diff_llm/create_dataset.py \
    --output-dir ./datasets/diff_corpus_small \
    --page-names '["Deep learning", "Ancient Greece", "Ted Chiang"]' \
    --n-revisions 1000
```

Medium dataset with all revisions for each page

```
python diff_llm/create_dataset.py \
    --output-dir ./datasets/diff_corpus_medium \
    --page-names '["Deep learning", "Ancient Greece", "Ted Chiang"]'
```

Large datasets with random sample of pages:

1,000 pages

```
python diff_llm/create_dataset.py \
    --output-dir ./datasets/diff_corpus_large1k \
    --n-random-pages 1000
```

10,000 pages

```
python diff_llm/create_dataset.py \
    --output-dir ./datasets/diff_corpus_large10k \
    --n-random-pages 10000
```

1,000,000 pages

```
python diff_llm/create_dataset.py \
    --output-dir ./datasets/diff_corpus_large1m \
    --n-random-pages 1000000
```

#### Zip Dataset

```
for d in datasets/*; do
    ./zip-dataset.sh ${d}
done
```

This will produce a tar gz file called `<dataset_path>.tar.gz`.

### Fine Tune

Train on xs dataset

```
python diff_llm/fine_tune.py \
    --model-path EleutherAI/pythia-70m \
    --data-dir=datasets/diff_corpus_xs \
    --output-dir=models/diff_model_xs
```

Train on medium dataset

```
# increase limit on open files
ulimit -n 1024

python diff_llm/fine_tune.py \
    --model-path EleutherAI/pythia-70m \
    --data-dir=datasets/diff_corpus_medium \
    --output-dir=models/diff_model_medium \
    --report-to wandb
```

### Inference

```
python diff_llm/inference.py \
    --model-path models/diff_model_medium \
    --tokenizer-path EleutherAI/pythia-70m \
    --prompt-file datasets/diff_corpus_xs/deep_learning_440439382_440443548.json
```


### Experiments

Prototype fine-tuning on the medium dataset
```
make fine-tuning-med-0
```

Padding the tokens to the left:
```
make fine-tuning-med-padding-left
```