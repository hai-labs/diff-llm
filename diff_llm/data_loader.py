"""Load data for training."""

import difflib
import json
import re
import typing
from pathlib import Path

from datasets import Dataset


class RawExample(typing.TypedDict):
    title: str
    before: str
    after: str
    revision_comment: str


class Example(typing.TypedDict):
    title: str
    context: str
    before: str
    after: str
    revision_comment: str


UNIFIED_DIFF = r"@@ [0-9,\+\- ]+ @@"


TEMPLATE = """
<TITLE>
{title}
</TITLE>

<CONTEXT>
{context}
</CONTEXT>

<BEFORE>
{before}
</BEFORE>

<REVISION_PROMPT>
{revision_comment}
</REVISION_PROMPT>

<AFTER>
{after}
</AFTER>
""".strip()


def parse_raw_example(raw_example: RawExample) -> typing.Iterator[Example]:
    diff = difflib.unified_diff(
        raw_example["before"].splitlines(),
        raw_example["after"].splitlines(),
        n=1,
    )
    difflist = [*diff]

    # remove the first two lines, which are the diff header
    diff_text = "\n".join(
        x for x in difflist[2:]
        if x != ""
        # this is some hacky logic... should really be cleaned up in the
        # data creation step
        and x != "-***************"
        and not re.match("\-\*\*\* [0-9,]+ \*\*\*\*", x)
    )
    diffs = [x for x in re.split(UNIFIED_DIFF, diff_text) if x != ""]
    for diff_lines in diffs:
        context, deleted, added = parse_diff_lines(diff_lines.split("\n"))
        yield Example({
            "title": raw_example["title"],
            "context": "\n".join(context),
            "before": "\n".join(deleted),
            "after": "\n".join(added),
            "revision_comment": raw_example["revision_comment"],
        })


def parse_diff_lines(diff_lines: list[str]) -> tuple[list[str]]:
    """
    Example diff:

    [
        "foo",   # context
        "-bar",  # deleted
        "+baz",  # added
        "",      # context (this won't be included)
    ]
    """
    context, deleted, added = [], [], []
    patt = []
    for line in diff_lines:
        if line == "":
            continue

        if not line.startswith("-") and not line.startswith("+"):
            if patt and patt[-1] in {"-", "+"}:
                break
            context.append(line)
        elif line.startswith("-"):
            deleted.append(line[1:])
            patt.append("-")
        elif line.startswith("+"):
            added.append(line[1:])
            patt.append("+")
        else:
            raise ValueError(f"Unexpected line in diff: {line}")
    
    return context, deleted, added


def iter_reader(data_dir: str) -> typing.Iterator[Example]:
    data_dir = Path(data_dir)
    count = 0
    for fp in data_dir.glob("*.json"):
        count += 1
        with fp.open("r") as f:
            data = json.load(f)
        yield from parse_raw_example(data)


def remove_large_diffs(example: Example, max_diff_len: int = 5000) -> bool:
    # large diffs tend to contain spam, or spam reversions.
    return (len(example["after"]) - len(example["before"])) <= max_diff_len


def format_example(example: Example) -> dict:
    return {"example": TEMPLATE.format(**example)}


def get_dataset(
    data_dir: str,
    seed: int = 43,
    remove_other_columns: bool = True,
) -> Dataset:
    dataset = Dataset.from_list([*iter_reader(data_dir)])
    map_kwargs = {"remove_columns": dataset.column_names} if remove_other_columns else {}
    return (
        dataset
        .filter(remove_large_diffs)
        .shuffle(seed=seed)
        .map(format_example, **map_kwargs)
    )


if __name__ == "__main__":
    dataset = get_dataset("datasets/diff_corpus_medium")
    for example in dataset:
        print(example["example"])
        break
