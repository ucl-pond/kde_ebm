# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('ebm')

    return config


def setup_package():
    metadata = dict(name='ebm',
                    maintainer='Nicholas C. Firth',
                    maintainer_email='nicholas.firth@ucl.ac.uk',
                    description='EBM code innit',
                    license='TBC',
                    url='',
                    version='0.0.1',
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

