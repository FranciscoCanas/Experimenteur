import numpy as np
import ConfigParser
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from tabulate import tabulate
from Experimenteur.dataset import Dataset
from Experimenteur.model import Model


class Experiment:
    """
    Represents the experimental procedure, including cross-validation methods. Must contain an instance of
     a dataset and model.
    """

    def __init__(self, config_path, model=None, dataset=None):
        self.configuration = ConfigParser.ConfigParser()
        self.configuration.read(config_path)
        self.properties = dict(self.configuration.items('experiment'))
        self.model = model if model else Model(dict(self.configuration.items('model')),
                                               dict(self.configuration.items('parameters')))
        self.dataset = dataset if dataset else Dataset(dict(self.configuration.items('dataset')))

        self.cross_val_fn_map = {
            'n_trials': self.n_trials_fn,
            'leave_one_out': self.leave_one_out_fn,
            'cv_split': self.cv_split_fn,
        }

    def run(self):
        """
        Run the cross validation loop and collect stats.
        """
        cv_fn = self.cross_val_fn_map(self.properties.get('cross_val_fn', 'cv_split'))
        folds = self.properties.get('folds', [''])

        for j, fold in enumerate(folds):
            X, y = self.dataset.load_data()
            self.statistics = []
            for X, X_held_out, y, y_held_out, in cv_fn(X, y):
                self.model.fit(X, y)
                self.statistics.append(self.model.evaluate(X_held_out, y_held_out))



    def display(self):
        """
        Display current experiment configuration arguments.
        """
        pass

    def cv_split_fn(self, X, y):
        balanced = self.properties.get('balanced', True)
        v_size = self.properties.get('validation_size', 0.25)
        shuffle_fn = StratifiedShuffleSplit if balanced else ShuffleSplit
        splits = shuffle_fn(y, test_size=v_size)
        return [(X[tinds], X[vinds], y[tinds], y[vinds]) for tinds, vinds in splits]

    def n_trials_fn(self):
        pass

    def leave_one_out_fn(self):
        pass




