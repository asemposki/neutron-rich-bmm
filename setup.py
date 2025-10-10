from distutils.core import setup, Extension

setup(
    name='neutron-rich-bmm',
    packages=['neutron-rich-bmm'],
    # cmdclass={'build_ext': CustomBuildExtCommand},
    # ext_modules=ext_modules,
    version='0.0.1',
    description='Gaussian process model mixing for the dense matter equation of state',
    author='Alexandra C. Semposki',
    author_email='as727414@ohio.edu',
    license='GPL-3.0',
    url='https://www.github.com/asemposki/neutron-rich-bmm',
    download_url='',
    keywords='BAND nuclear physics model mixing gaussian process uncertainty quantification dense matter',
    classifiers=[
        'Development Status :: 1 - Beta',
        'License :: OSI Approved :: GPL-3.0 License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific :: Physics',
        'Topic :: Scientific :: Statistics'
        ],
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'Cython>=0.29',
        'numdifftools',
        'scikit-learn',
    ]
)
