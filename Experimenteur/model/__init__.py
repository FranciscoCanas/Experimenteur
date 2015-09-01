import importlib
from tabulate import tabulate


class Model:
    """
    Represents a statistical model. Must implement a function used to fit/train, and a function
    used to evaluate.
    """

    def __init__(self, properties, parameters):
        self.properties = properties
        self.name = self.properties.get('name', 'name')
        self.init_params(parameters)
        self.init_model()

    def init_params(self, parameters):
        self.parameters = {}
        for k, v in parameters.items():
            v = eval(v) if isinstance(v, str) else v
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
        module_name, class_name = model_class.rsplit(".", 1)
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
            acc = self.model.score(X, y)
            metrics.append(acc)

        return metrics

    def evaluate(self, X, y):
        """
        This method assumes your model instance uses sklearn style interface: ie. score(X,y).

        Override if it doesn't.
        :return: Array of metrics.
        """
        metrics = []
        acc = self.model.score(X, y)
        metrics.append(acc)
        return metrics

    def display(self):
        print 'Model:'
        print tabulate([(k, str(v)) for k, v in self.properties.iteritems()])
        print 'Parameters:'
        print tabulate([(k, str(v)) for k, v in self.parameters.iteritems()])




