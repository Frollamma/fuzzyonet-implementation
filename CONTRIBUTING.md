# Contribution guide

...

## Roadmap

- [x] ANFIS inference ([here](./src/fuzzyonet/fuzzyonet_from_scratch.py))
- [x] Tests for ANFIS inference ([here](./tests/anfis_function.py))
- [] ANFIS training with tests
- [] ANFIS functional inference with tests
- [] ANFIS functional training with tests
- [] ANFIS operator (FuzzyONet) inference with tests
- [] ANFIS operator (FuzzyONet) training with tests

## TODOs

- Many `TODO`s in the repo...
- Problems with `scikit-anfis`
  - Missing dependencies: `scikit-learn` (fixed in [our fork](https://github.com/Frollamma/scikit-anfis))
  - This library does things I wouldn't do/I don't understand, like rounding results, see [this script](./tests/understand_scikit_anfis.py)
  - They didn't create a package on pypi
  - It loads slowly, here are some benchmarks for comparison, the scripts just import the libraries and print an hello world

    ```
    $ time python tests/hello/hello_sklearn.py
    Hello from fuzzyonet!

    ________________________________________________________
    Executed in  707.72 millis    fish           external
      usr time    1.98 secs      0.28 millis    1.98 secs
      sys time    0.51 secs      2.03 millis    0.51 secs

    $ time python tests/hello/hello_xanfis.py
    Hello from fuzzyonet!

    ________________________________________________________
    Executed in    2.54 secs    fish           external
      usr time    3.66 secs   10.16 millis    3.65 secs
      sys time    0.69 secs    2.14 millis    0.69 secs

    $ time python tests/hello/hello_skanfis.py
    Hello from fuzzyonet!

    ________________________________________________________
    Executed in    4.20 secs    fish           external
      usr time    4.79 secs    0.00 micros    4.79 secs
      sys time    0.69 secs  726.00 micros    0.69 secs
    ```
