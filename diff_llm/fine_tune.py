"""Fine-tune a pre-trained LLM on diff-completion task."""

from functools import partial

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data_loader import get_dataset


def fine_tune(data_dir: str):
    dataset = get_dataset(data_dir)
    import ipdb; ipdb.set_trace()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")


if __name__ == "__main__":
    fine_tune("datasets/diff_corpus_xs")
