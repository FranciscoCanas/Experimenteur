import tables as tb
import numpy as np
import pandas as pd
import h5py as h5

class Dataset:
    """
    Represents a dataset used in an experiment. ie:
    dataset_args = {
    'ADNI_Cortical_Features': {
        'source_path': source_path,
        'class_names': ['ad', 'cn', 'mci'],
        'load_fn': load_matrices,
        'dataset': 'ADNI_Cortical_Features',
        'omit_class': None, # ie: Omit ad, classifies between cn and mci
        'use_fused': False,
        'variables': 'all',
        'balance': False,
    },
    'HC': {
        'source_path': source_path,
        'class_names': ['ad', 'cn', 'mci'],
        'load_fn': load_matrices,
        'dataset': 'hc',
        'omit_class': None,
        'structure': 'hc',
        'side': 'b',
        'use_fused': True,
    },
    'multiple': {
        'dataset': 'multiple',
        'target': 'ADAS13',
        'load_fn': load_multiple_modalities,
    }
}
    """

    def __init__(self, properties):
        self.properties = properties
        if not self.properties.has_key('source_path'):
            raise Exception('Missing \'source_path\' property in dataset configuration options.')

        self.load_fns = {
            'npy': np.load,
            'tables': tb.open_file,
            'txt': np.loadtxt,
            # TODO: Add more file formats
        }

    def load_data(self):
        """
        This method should be overriden. Base version assumes basic data format.
        - load_data: Return an experimental numpy matrix, andoptional  vector of labels.

        dataset options required:
        - 'format' for non-.npy format data.
        """
        format = self.properties.get('format', 'npy')
        if not format in self.load_fns:
            raise Exception('Unsupported \'format\' property in dataset configuration options.')

        load_fn = self.load_fns[format]

        X = load_fn(self.properties.get('source_path'))

        return X

    def load_test_data(self):
        """
        This method should be overridden: Return a numpy matrix and optional vector of labels.
        """
        pass


