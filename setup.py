# Authors: Nicholas C. Firth, Neil P. Oxtoby
# License: MIT
__version__ = '0.0.3'

def setup_package():
    from setuptools import setup, find_packages
    metadata = dict(name='kde_ebm',
                    maintainer='Neil P. Oxtoby',
                    description='Event Based Model with Kernel Density Estimation mixture modelling',
                    license='MIT',
                    url='https://ucl-pond.github.io/kde_ebm',
                    version=__version__,
                    zip_safe=False,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Programming Language :: Python',
                                 'Topic :: Scientific/Engineering',
                                 'Programming Language :: Python :: 2',
                                 'Programming Language :: Python :: 2.7',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.6',
                                 ],
                    packages=find_packages()
                    )

    install_requires = [
        "pybind11",
        "scipy>=0.9",
        "numpy>=1.6.1",
        "matplotlib>=2.0.1",
        "tqdm",
        "awkde @ git+https://github.com/noxtoby/awkde.git",
        "scikit-learn"
    ]
    metadata['install_requires'] = install_requires
    
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
