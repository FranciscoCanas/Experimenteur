from distutils.core import setup

setup(
        name='Experimenteur',
        version='0.0.0.1',
        author='Francisco Canas',
        author_email='mailfrancisco@gmail.com',
        maintainer='Francisco Canas',
        maintainer_email='mailfrancisco@gmail.com',
        packages=[
            'Experimenteur',
            'Experimenteur.dataset',
            'Experimenteur.experiment',
            'Experimenteur.model',
            'Experimenteur.test',
            'Experimenteur.visualizations',
        ],
        data_files=[
            ('data', ['data/cfg/*.cfg']),
            ('data', ['data/frames/*.csv']),
            ('data', ['data/matrices/*.npy']),
        ],
        url='https://github.com/franciscocanas/experimenteur',
        license='LICENSE.txt',
        description='A tool for automating exploratory data analysis with with scikit-learn.',
        long_description=open('README.md').read(),
        install_requires=[
                "scikit-learn",
                "matplotlib",
                "h5py",
                "pandas",
                "tabulate",
                "tables",
            ]
        )
