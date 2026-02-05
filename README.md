# FuzzyONet

We give an implementation of FuzzyONet as in described in [FuzzyONet paper](TODO_missing_link)

## Install

If you have [uv](https://docs.astral.sh/uv/) installed run

```sh
uv sync
```

Otherwise, you will need to install the deps manually or change the `pyproject.toml` file.

## Run

- Enter the python virtual environment
- Run

```sh
jupyter notebook main.ipynb
```

- Check your browser, a new window should appear

## TODOs

Problems with `scikit-anfis`

- Missing dependencies: `scikit-learn`
- They didn't create a package on pypi
- It loads slowly, here are some benchmarks for comparison, the scripts just import the libraries and print an hello world

  ```
  $ time python extra/hello_sklearn.py
  Hello from fuzzyonet!

  ________________________________________________________
  Executed in  707.72 millis    fish           external
    usr time    1.98 secs      0.28 millis    1.98 secs
    sys time    0.51 secs      2.03 millis    0.51 secs

  $ time python extra/hello_xanfis.py
  Hello from fuzzyonet!

  ________________________________________________________
  Executed in    2.54 secs    fish           external
    usr time    3.66 secs   10.16 millis    3.65 secs
    sys time    0.69 secs    2.14 millis    0.69 secs

  $ time python extra/hello_skanfis.py
  Hello from fuzzyonet!

  ________________________________________________________
  Executed in    4.20 secs    fish           external
    usr time    4.79 secs    0.00 micros    4.79 secs
    sys time    0.69 secs  726.00 micros    0.69 secs
  ```

The license is MIT, so you can fork and fix the problems
