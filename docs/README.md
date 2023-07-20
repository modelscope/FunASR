# FunASR document generation

## Generate HTML
For convenience, we provide users with the ability to generate local HTML manually.

First, you should install the following packages, which is required for building HTML:

```sh
pip3 install -U "funasr[docs]"
```

Then you can generate HTML manually.

```sh
cd docs
make html
```

The generated files are all contained in the "FunASR/docs/_build" directory. You can access the FunASR documentation by simply opening the "html/index.html" file in your browser from this directory.