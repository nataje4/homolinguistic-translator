## ABOUT

This is a simple implementation of my homolinguistic translator tool, meant to be run from the command line. It uses files generated by Stanford NLP's [GloVe](https://github.com/stanfordnlp/GloVe) machine learning model to produce word-by-word "translations" of the text that you provide. I have provided the GloVe information files that I have generated by scraping the Poetry Foundation's website, but you can provide your own files if you want to. Just use the optional CLI flags `--vectors_file` and `--vocab_file`. If those flags aren't provided, the program will default to using the Poetry Foundation corpus.

## QUICKSTART

1. [Create and source a virtual environment](https://docs.python.org/3/library/venv.html).
2. Install project requirements: 
```
pip3 install -r requirements.txt
```
3. Run the program, either interactively (`python3 main.py`) or using the input file flag to translate a large amount of text (`python3 main.py --input-file=path/to/input/file`). If you don't provide an input file, you'll be able to type your input directly into the shell.