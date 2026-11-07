import contextlib
import io
import os
import re

README = os.path.join(os.path.dirname(__file__), "..", "README.md")


def test_the_example_of_the_readme_prints_what_the_readme_says():
    text = open(README, encoding="utf-8").read()
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    expected = re.search(r"```python\n.*?```\n\n```\n(.*?)```", text, re.S).group(1)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README.md", "exec"), {"__name__": "__readme__"})
    assert out.getvalue() == expected
