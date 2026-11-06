# Nastaliq-Engine
I had difficulty getting resources to learn urdu. That is when I realised the OCR quality isnt great. Searching is really hard, with such a complex written language such as urdu, this is gonna be a challenge but smth im willing to undertake...

## Layout

```
src/     pipeline: normalize -> build_corpus -> render -> augment -> sequence -> model -> ctc
tests/   unit tests
misc/    previews and benchmarks
data/    corpus and render cache (git-ignored, except charset.json)
```

## Setup

```bash
sudo apt install python3-pil fonts-noto-core    # Pillow with libraqm, Noto Nastaliq Urdu
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

## Running

Everything runs from the repo root as a module:

```bash
python -m src.build_corpus          # data/corpus.tsv, data/charset.json
python -m src.cache                 # clean render of every line
python -m misc.preview_augment      # out/augment_sheet.png
python -m unittest discover -s tests -t .
```



