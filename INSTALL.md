Installation: KDE Event-Based Model
=================

As of 2021-05-19, the following should work:

`pip install git+https://github.com/ucl-pond/kde_ebm`

(If not, try also this: `pip install git+https://github.com/ucl-pond/kde_ebm --use-feature=in-tree-build --force-reinstall`)


<hr/>

## Old instructions

You'll need `pip` and `virtual_env`.

## Setup a virtual environment 

Open a terminal, install `virtualenv` using pip, create a new virtual environment with python 3:

```
pip install virtualenv
```

```
virtualenv kdeebm_env --python=python3
```

Activate the virtual environment:

* macOS/linux:
  ```javascript macOS/linux
  source kdeebm_env/bin/activate
  ```
* Windows:
  ```javascript Windows
  kdeebm_env\Scripts\activate
  ```

You are now in a virtual environment. It's probably less exciting than it sounds, but it's important.

## Install Dependencies

See below for how to use the provided `requirements.txt` file to install the following dependencies via `pip`:
- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [awkde](https://github.com/noxtoby/awkde)
- [Matplotlib](https://github.com/matplotlib/matplotlib)

Use `pip` to install dependencies
```
pip install -r requirements.txt
```

**NOTE**: On macOS, if you encounter errors above, then it might be an issue with Command Line Tools. Best advice: repeat the virtual environment setup, but be sure to use homebrew's version of python3:

```
brew install python3
/usr/local/bin/python3 -m venv kdeebm_env 
/usr/local/bin/virtualenv kdeebm_env --python=python3
source kdeebm_env/bin/activate
```
then install dependencies using:
```
/usr/local/bin/python3 -m pip install -r requirements.txt
```



You should be good to go! Go ahead and play with the [examples](examples).

If using a jupyter notebook, be sure to call jupyter from within your environment so that you're using the correct python installation. `<sarcasm>` *Hashtag fun times* `</sarcasm>`
