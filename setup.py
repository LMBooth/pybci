"""Python setup script for the pybci distribution package."""

from setuptools import setup, find_packages
from codecs import open
from os import path
import sys


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = not sys.platform.startswith("win")
        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            # We don't contain any python source
            python, abi = 'py3', 'none'
            return python, abi, plat
except ImportError:
    bdist_wheel = None


here = path.abspath(path.dirname(__name__))


# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# Get the version number from the version file
# Versions should comply with PEP440.  For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
version = {}
with open("pybci/version.py") as fp:
    exec(fp.read(), version)


setup(
    name='pybci-package',

    version=version['__version__'],

    description='A Python interface to create a BCI with the Lab Streaming Layer, Pytorch, SciKit-Learn and Tensorflow packages',
    long_description=long_description,
    long_description_content_type="text/markdown",

    # The project's main homepage.
    url='https://github.com/lmbooth/pybci',

    # Author details
    author='Liam Booth',
    author_email='liambooth123@hotmail.co.uk',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',

        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',
    # What does your project relate to?
    keywords='machine-learning tensorflow sklearn pytorch human-computer-interaction bci lsl brain-computer-interface labstreaminglayer',

    cmdclass={
        'bdist_wheel': bdist_wheel
    },

    install_requires=[
        "numpy>=1.21,<1.26",
        "pylsl>=1.16.1",
        "scipy>=1.11.1",
        "antropy>=0.1.6",
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.1",
    ],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages= find_packages(),#['pybci', 'pybci.examples'],
    entry_points={
        'console_scripts': [
            'pybci = pybci.cli:main',
        ],
    },
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # install_requires=[]

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={},

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # Here we specify all the shared libs for the different platforms, but
    # setup will probably only find the one library downloaded by the build
    # script or placed here manually.
    package_data={},
    
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={},
    #
    # ext_modules=extension_modules,
)
