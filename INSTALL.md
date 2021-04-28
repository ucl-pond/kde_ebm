Installation: KDE Event-Based Model
=================

You'll need `pip`.

Setup a virtual environment 
============

Open a terminal, install `virtualenv` using pip, create a new virtual environment:

```
pip install virtualenv
```

```
virtualenv kdeebm_env --python=python3.9
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

Install Dependencies
============
- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [awkde](https://github.com/noxtoby/awkde)
- [Matplotlib](https://github.com/matplotlib/matplotlib)

On macOS, if you encounter errors below, then it might be an issue with Command Line Tools. Best advice: repeat the virtual environment setup, but be sure to use homebrew's version of python3:

```
brew install python3
/usr/local/bin/python3 -m venv kdeebm_env 
/usr/local/bin/virtualenv kdeebm_env --python=python3.9
source kdeebm_env/bin/activate
```


Use `pip` to install dependencies

```
pip install -r requirements.txt
```
or
```
/usr/local/bin/python3 -m pip install -r requirements.txt
```

You are good to go! Go ahead and play with the [examples](examples).
