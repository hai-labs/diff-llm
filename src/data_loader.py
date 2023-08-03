"""Load data for training."""

import difflib
import json
from pathlib import Path


def main(data_dir: str):
    data_dir = Path(data_dir)
    count = 0
    for fp in data_dir.glob("*.json"):
        count += 1
        with fp.open("r") as f:
            data = json.load(f)

        print("-----------------")
        print("BEFORE\n\n", data["before"])
        print("REVISION_COMMENT\n\n", data["revision_comment"])
        print("AFTER\n\n", data["after"])
        print("DIFF\n\n")
        for line in difflib.unified_diff(
            data["before"].splitlines(), data["after"].splitlines()
        ):
            print(line)


if __name__ == "__main__":
    main("datasets/diff_corpus_xs")
