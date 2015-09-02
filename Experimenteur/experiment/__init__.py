import numpy as np
import ConfigParser
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit, KFold
from tabulate import tabulate
from Experimenteur.dataset import Dataset
from Experimenteur.model import Model


class Experiment:
    """
    Represents the experimental procedure, including cross-validation methods. Must contain an instance of
     a dataset and model.
    """
    statistics_header = ['Training Score', 'Validation Score']
    metrics_header = []
    summary_header = ['Training Score Mean', 'STD', 'Validation Score Mean', 'STD']
    properties_header = ['Property', 'Value']

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
        cv_fn = self.cross_val_fn_map[self.properties.get('cross_val_fn', 'cv_split')]
        folds = self.properties.get('folds', [''])

        for j, fold in enumerate(folds):
            self.dataset.load_data(fold)
            self.metrics = []

            for X, X_held_out, y, y_held_out, in cv_fn():
                training_metrics = self.model.fit(X, y)
                validation_metrics, header = self.model.score(X_held_out, y_held_out)

                self.metrics.append(training_metrics + validation_metrics)
                self.metrics_header = header

            self.metrics = np.array(self.metrics)
            self.summary()


    def report(self):
        print
        print tabulate(self.metrics, self.statistics_header + self.metrics_header)
        print
        print tabulate(self.summary_stats, self.summary_header)

    def summary(self):
        trains = self.metrics[:,0]
        valids = self.metrics[:,1]
        self.summary_stats = [[np.mean(trains), np.std(trains), np.mean(valids), np.std(valids)]]


    def display(self):
        """
        Display current experiment configuration arguments.
        """
        print
        print 'Experimental Parameters:'
        print tabulate(self.properties.items(), self.properties_header)
        self.model.display()
        self.dataset.display()


    def cv_split_fn(self):
        X = self.dataset.X
        y = self.dataset.y
        balanced = self.properties.get('balanced', True)
        folds = int(self.properties.get('cv_folds', 1))
        v_size = float(self.properties.get('validation_size', 0.25))
        if balanced and 'classification' in self.model.type:
            splits = StratifiedShuffleSplit(y, n_iter=folds, test_size=v_size)
        else:
            splits = ShuffleSplit(X.shape[0], test_size=v_size, n_iter=folds)

        return [(X[tinds], X[vinds], y[tinds], y[vinds]) for tinds, vinds in splits]

    def n_trials_fn(self):
        n = self.properties.get('n_trials', 10)
        return [(self.dataset.X, self.dataset.y, self.dataset.X_v, self.dataset.y_v) for i in range(n)]

    def leave_one_out_fn(self):
        X = self.dataset.X
        y = self.dataset.y
        N = X.shape[0]
        splits = KFold(N, n_folds=N, random_state=0)
        return [(X[tinds], X[vinds], y[tinds], y[vinds]) for tinds, vinds in splits]




