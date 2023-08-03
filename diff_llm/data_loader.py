"""Load data for training."""

import json
import typing
from functools import partial
from pathlib import Path

from datasets import Dataset


class Example(typing.TypedDict):
    title: str
    before: str
    after: str
    revision_comment: str


TEMPLATE = """
<title>{title}</title>
<before>{before}</before>
<revision_comment>{revision_comment}</revision_comment>
<after>{after}</after>
""".strip()


def iter_reader(data_dir: str) -> typing.Iterator[Example]:
    data_dir = Path(data_dir)
    count = 0
    for fp in data_dir.glob("*.json"):
        count += 1
        with fp.open("r") as f:
            yield json.load(f)


def format_example(example: Example) -> dict:
    return {"example": TEMPLATE.format(**example)}


def get_dataset(data_dir: str, seed: int = 43) -> Dataset:
    dataset = Dataset.from_generator(partial(iter_reader, data_dir))
    return (
        dataset.shuffle(seed=seed)
        .map(format_example, remove_columns=dataset.column_names)
    )


if __name__ == "__main__":
    dataset = get_dataset("datasets/diff_corpus_xs")
    for example in dataset:
        print(example)
        break
