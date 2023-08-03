"""Get data for diff prediction task.

The task is, given target line(s) t and surrounding context c, predict the t'
which are the set of tokens that replace t in the next revision.

For example:

c = "The quick brown fox\njumps over the lazy dog\nand then went home."
t = "jumps over the lazy dog"
t' = "jumped over the lazy dogs"

NOTE:
Simplification: Just use the context diff representation:
https://docs.python.org/3/library/difflib.html#difflib.context_diff

```
doc = '''
<before>
The quick brown fox
jumps over the lazy dog
and then went home.
</before>

<revision-comment>
Make past tense.
</revision-comment>

<after>
The quick brown fox
jumped over the lazy dogs
and then went home.
</after>
'''
```
"""

import difflib
import itertools
import json
import logging
import re
import typing
from pathlib import Path

import pywikibot


DIFF_HEADER = r"\*\*\* rev_\d+\s--- rev_\d+\s"

# changed content is delimited by:
# 
# *** rev_<i>
# --- rev_<j> ***
# ***************
# *** <n>,<m> ****
DIFF_DELIMITER = (
    r"\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\s"
    r"\*\*\* [0-9,]+ \*\*\*\*"
)


logger = logging.getLogger(__name__)


class DocDiff(typing.NamedTuple):
    before: str
    after: str
    diff_text: str

    def __str__(self) -> str:
        return (
            "📖 DocDiff:\n\n"
            f"⏸️ Before:\n{self.before}\n\n"
            f"⏩️ After:\n{self.after}\n\n"
            f"📝 Diff text:\n{self.diff_text}\n\n"
        )
    

class PageDiff(typing.NamedTuple):
    title: str
    doc_diff: DocDiff
    revision_comment: str
    old_revid: str
    new_revid: str


def get_raw_diff(
    page,
    old_rev: str,
    new_rev: str,
    n_context_lines: int,
) -> typing.Optional[list[str]]:
    """Get diff between two revisions.

    Args:
        old_rev (str): Old revision id.
        new_rev (str): New revision id.

    Returns:
        str: Diff string.
    """
    # TODO: handle new_text is none
    new_text = page.getOldVersion(oldid=new_rev)
    if new_text is None:
        logging.info(f"Couldn't find text for new revision {new_text}")
        return None

    old_text = page.getOldVersion(oldid=old_rev)
    if old_text is None:
        logging.info(f"Couldn't find text for old revision {old_text}")
        return None

    unified_diff = difflib.context_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        lineterm="",
        n=n_context_lines,
        fromfile=f'rev_{old_rev}',
        tofile=f'rev_{new_rev}',
    )

    return list(unified_diff)
    

def parse_diffs(diffs: list[str]) -> typing.Iterator[DocDiff]:
    assert len(diffs) > 0, "diff cannot be empty"
    diffs_text: str = "\n".join(diffs)
    diffs_text = re.sub(DIFF_HEADER, "", diffs_text)
    diffs_text = re.split(DIFF_DELIMITER, diffs_text)

    for d in diffs_text:
        if not d:
            continue
        try:
            yield create_doc_diff(d)
        except Exception as exc:
            logger.error(f"Failed to parse diff:\n{d} - {exc}")


def parse_diff_lines(before: str, after: str) -> tuple[str, str]:
    before_out, after_out = [], []
    for l1, l2 in itertools.zip_longest(
        (x for x in before.splitlines() if x.strip()),
        (x for x in after.splitlines() if x.strip()),
        fillvalue="",
    ):
        if l1.strip() == "" and l2.strip() == "":
            continue

        if not before:
            # no content in the before text means that only lines were removed
            # from the after text.
            if l2.startswith("+"):
                after_out.append(l2)
            else:
                after_out.append(l2)
                before_out.append(l2)
            continue
    
        if not after:
            # no content in the after text means that only lines were removed
            # from the before text.
            if l1.startswith("-"):
                before_out.append(l1)
            else:
                before_out.append(l1)
                after_out.append(l1)
            continue
        
        before_out.append(l1)
        after_out.append(l2)

    return before_out, after_out


def create_doc_diff(diff_text: str) -> DocDiff:
    # split diff_doc by regex
    # before and after is delimited by --- <n>,<m> ----
    split = re.split(r"--- [0-9,]+ ----", diff_text)
    before, after = split
    before, after = parse_diff_lines(before.strip(), after.strip())

    # the first two characters of each line are the diff indicator, e.g.
    # "- ", "+ ", "! ", or "  " for no change.
    def clean_text(text: list[str]):
        return "\n".join(
            x[2:] if x.startswith(("+", "-", "!")) else x for x in text
        ).strip()

    return DocDiff(clean_text(before), clean_text(after), diff_text)


def process_page(
    site: pywikibot.Site,
    page_name: str,
    output_dir: Path,
    n_revisions: typing.Optional[int] = None,
    n_context_lines: int = 2,
    use_cache: bool = True,
) -> typing.Iterator[PageDiff]:
    page = pywikibot.Page(site, page_name)
    revisions = [*page.revisions(reverse=True, total=n_revisions)]
    logger.info(f"Number of revisions: {len(revisions)}")

    for old_rev, new_rev in zip(revisions[:-1], revisions[1:]):
        fp = get_doc_file_name(output_dir, page_name, old_rev['revid'], new_rev['revid'])
        if use_cache and fp.exists():
            logging.info(f"Data point {fp} already exists: skipping raw diff processing.")
            continue
        logging.info(f"Page: {page_name} - Revision: {old_rev['revid']} -> {new_rev['revid']}")
        diff = get_raw_diff(
            page,
            old_rev["revid"],
            new_rev["revid"],
            n_context_lines,
        )
        if diff is None:
            logging.info(f"Text not found, skipping diff.")
            continue
        if len(diff) > 0:
            for diff_doc in parse_diffs(diff):
                yield PageDiff(
                    page.title(),
                    diff_doc,
                    new_rev["comment"],
                    old_rev['revid'],
                    new_rev['revid'],
                )


def get_doc_file_name(
    output_dir: Path,
    page_name: str,
    old_revid: str,
    new_revid: str,
) -> Path:
    page = page_name.lower().replace(' ', '_')
    return (output_dir / f"{page}_{old_revid}_{new_revid}").with_suffix(".json")


def main(
    page_names: list[str],
    output_dir: str,
    n_revisions: typing.Optional[int] = None,
    n_context_lines: int = 2,
    use_cache: bool = True,
):
    site = pywikibot.Site(u"en", fam=u"wikipedia")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_name in page_names:
        logging.info(f"Processing page: {page_name}")
        for page_diff in process_page(
            site,
            page_name,
            output_dir,
            n_revisions=n_revisions,
            use_cache=use_cache,
            n_context_lines=n_context_lines,
        ):
            fp = get_doc_file_name(
                output_dir, page_name, page_diff.old_revid, page_diff.new_revid,
            )
            with fp.open("w") as f:
                data = {
                    **page_diff.doc_diff._asdict(),
                    "title": page_diff.title,
                    "revision_comment": page_diff.revision_comment,
                }
                json.dump(data, f)

        # TODO:
        # - Exclude documents with small diffs (need to determine threshold).
        # - Analyze and spot-check documents to see if data is correctly formatted.


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--page-names", type=str, required=True)
    parser.add_argument("--n-revisions", type=int, required=False, default=None)
    parser.add_argument("--n-context-lines", type=int, required=False, default=2)
    parser.add_argument("--use-cache", action="store_true", required=False)

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(
        json.loads(args.page_names),
        output_dir=args.output_dir,
        use_cache=args.use_cache,
        n_context_lines=args.n_context_lines,
        n_revisions=args.n_revisions,
    )
