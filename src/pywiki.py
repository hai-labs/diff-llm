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

<after>
The quick brown fox
jumped over the lazy dogs
and then went home.
</after>
'''
```

Simple case
-----------

One document in this dataset would look like:

doc = '''
<context>
The quick brown fox
jumps over the lazy dog
and then went home.
</context>

<diff>
<replace>
jumps over the lazy dog
<with>
jumped over the lazy dogs
</replace>
</diff>
'''

Multiple line changes
---------------------

A diff context that contains multiple contiguous line changes are treated
together:

```
c = "The quick brown fox\njumps over the lazy dog\nand then went home."
t = "jumps over the lazy dog\nand then went home."
t' = "jumped over the lazy dogs\nand then ran away."
```

doc = '''
<context>
The quick brown fox
jumps over the lazy dog
and then went home.
</context>

<diff>
<replace>
jumps over the lazy dog
<with>
jumped over the lazy dogs
</replace>

<replace>
and then went home.
<with>
and then ran away.
</replace>
</diff>
'''

If the context window is 0, it means that all lines in the context are part of
the diff.

Appending/preprending and truncating
------------------------------------

To handle cases where the diff involves adding or removing lines at the beginning
or end of a context, we add a special token to <diff> tag:

The following is a diff for deleting the beginning of the context:

'''
<context>
The quick brown fox
jumps over the lazy dog
and then went home.
</context>

<diff>
<truncate-beginning>
The quick brown fox
</truncate-beginning>
</diff>
'''

Or the end of a context:

```
<diff>
<truncate-end>
and then went home.
</truncate-end>
</diff>
```

Similarly, adding text to the beginning:

```
<diff>
<prepend>
Let me tell you something.
</prepend>
</diff>
```

and to the end:

```
<diff>
<append>
The end!
</append>
</diff>
```

Or a combination of the two:

```
<diff>
<prepend>
Let me tell you something.
</prepend>

<truncate-end>
and then went home.
</truncate-end>
</diff>
```

Or even the three transformations:

```
<diff>
<prepend>
Let me tell you something.
</prepend>

<replace>


<truncate-end>
and then went home.
</truncate-end>
</diff>
```
"""

import difflib

from pprint import pprint
import pywikibot


def parse_diffs():
    ...


def process_page(site: pywikibot.Site, page: str):
    page = pywikibot.Page(site, page)

    revisions = [*page.revisions(reverse=True)]
    print(f"Number of revisions: {len(revisions)}")

    for i, (old_rev, new_rev) in enumerate(zip(revisions[:-1], revisions[1:])):

        new_id, old_id = new_rev["revid"], old_rev["revid"]
        new_text = page.getOldVersion(oldid=new_id)
        old_text = page.getOldVersion(oldid=old_id)

        unified_diff = difflib.context_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            lineterm="",
            n=4,
            fromfile=f'rev_{old_id}',
            tofile=f'rev_{new_id}',
        )

        unified_diff_list = list(unified_diff)
        print("")
        print(f"Diff: {old_id} -> {new_id}")
        print("\n".join(unified_diff_list))
        print("------------------------")
        print("")
        if i > 20:
            break


def main(page_names: list[str]):
    site = pywikibot.Site(u"en", fam=u"wikipedia")
    for page_name in page_names:
        process_page(site, page_name)

        # TODO:
        # - Parse the unified diff to create diff documents as described above.
        # - Exclude documents with small diffs (need to determine threshold).
        # - Create small dataset of documents based on some pre-determined pages.
        # - Analyze and spot-check documents to see if data is correctly formatted.


if __name__ == "__main__":
    page_names = ["Deep learning"]
    main(page_names)
