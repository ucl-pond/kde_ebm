def setup_package():
    from setuptools import setup, find_packages
    metadata = dict(name='kde_ebm',
                    packages=find_packages()
                    )
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
