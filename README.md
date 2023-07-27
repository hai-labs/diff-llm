# DiffLLM

Extending the next token prediction setting into "diff prediction".

## Setup

Create virtual environment:

```
conda create -n diff-llm python=3.9
conda activate diff-llm
pip install -r requirements.txt
```

Export secrets:

```
export $(grep -v '^#' secrets.txt | xargs)
```
