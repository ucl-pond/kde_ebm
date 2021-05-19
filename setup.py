# Authors: Nicholas C. Firth, Neil P. Oxtoby
# License: TBC
__version__ = '0.0.2'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('kde_ebm')

    return config


def setup_package():
    metadata = dict(name='kde_ebm',
                    maintainer='Neil P. Oxtoby',
                    maintainer_email='n.oxtoby@ucl.ac.uk',
                    description='KDE EBM code',
                    license='TBC',
                    url='',
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
                    )

    from numpy.distutils.core import setup

    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()

