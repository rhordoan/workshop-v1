import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path


def create_sample_notebook(path: str = "00_scratchpad.ipynb") -> Path:
    """
    Create a simple Jupyter notebook with a couple of example cells.

    Run this script (python test.py) and then open the generated
    00_scratchpad.ipynb file in Jupyter / VS Code / Cursor.
    """
    nb = new_notebook(
        cells=[
            new_markdown_cell(
                "# Sample Notebook\n\n"
                "This is a small example Jupyter notebook generated from Python."
            ),
            new_code_cell(
                "# A simple Python cell\n"
                "x = 1 + 1\n"
                "x"
            ),
            new_markdown_cell(
                "You can add more cells, plots, and explanations as needed."
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
    )

    target = Path(path).resolve()
    target.write_text(nbformat.writes(nb), encoding="utf-8")
    return target


if __name__ == "__main__":
    notebook_path = create_sample_notebook()
    print(f"Sample notebook created at: {notebook_path}")









