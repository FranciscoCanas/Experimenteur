import importlib
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from tabulate import tabulate


class Model:
    """
    Represents a statistical model. Must implement a function used to fit/train, and a function
    used to evaluate.
    """

    def __init__(self, properties, parameters):
        self.properties = properties
        self.name = self.properties.get('name', 'myexperiment')
        self.init_params(parameters)
        self.init_model()
        self.type = self.properties.get('type', 'classification')

    def init_params(self, parameters):
        # TODO: Rethink this whole hack
        self.parameters = {}
        for k, v in parameters.items():
            try:
                v = eval(v)
            except (TypeError, NameError) as e:
                if type(v) is str and 'true' is v.lower() or 'false' is v:
                    v = bool(v)
            self.parameters[k] = v

    def init_model(self):
        """
        Instantiate the model:
        ie. sklearn.ensemble.AdaBoost

        Override this method when building custom models.
        """
        model_class = self.properties.get('class')
        if not model_class:
            raise Exception('No model class specified under model properties.')
        module_name, class_name = model_class.rsplit('.', 1)
        class_ = getattr(importlib.import_module(module_name), class_name)
        self.model = class_(**self.parameters)


    def fit(self, X, y=None):
        """
        This method assumes your model instance uses sklearn style interface: ie. fit(X,y).

        Override if it doesn't.
        :return: Array of training metrics.
        """
        metrics = []
        if y is None:
            self.model.fit(X)

        else:
            self.model.fit(X, y)
            score = self.model.score(X, y)
            metrics.append(score)

        return metrics

    def score(self, X, y=None):
        """
        This method assumes your model instance uses sklearn style interface: ie. score(X,y).

        Override if it doesn't.
        :return: Array of metrics.
        """
        metrics_arr = []
        score = self.model.score(X, y)
        metrics_arr.append(score)
        m, h = self.metrics(X, y)
        metrics_arr += m
        return metrics_arr, h

    def metrics(self, X, y=None):
        """

        :return:
        """
        m = []
        h = []
        y_hat = self.model.predict(X)

        if 'classification' in self.type:
            m.append(sklearn.metrics.accuracy_score(y, y_hat))
            m.append(sklearn.metrics.precision_score(y, y_hat))
            m.append(sklearn.metrics.recall_score(y, y_hat))
            m.append(sklearn.metrics.f1_score(y, y_hat))
            h = ['acc', 'prec', 'rec', 'f1']
        elif 'regression' in self.type:
            m.append(sklearn.metrics.explained_variance_score(y, y_hat))
            m.append(sklearn.metrics.r2_score(y, y_hat))
            h = ['Evar', 'r2']
        return m, h


    def display(self):
        print
        print 'Model:'
        print tabulate([(k, str(v)) for k, v in self.properties.iteritems()])
        print 'Parameters:'
        print tabulate([(k, str(v)) for k, v in self.parameters.iteritems()])




