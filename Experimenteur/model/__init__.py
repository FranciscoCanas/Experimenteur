import importlib

class Model:
    """
    Represents a statistical model. Must implement a function used to fit/train, and a function
    used to evaluate.
    """

    def __init__(self, properties, parameters):
        self.properties = properties
        self.parameters = parameters
        self.name = self.properties.get('name', 'name')
        self.init_model()

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


    def fit(self):
        """
        This method assumes your model instance uses sklearn style interface: ie. fit(X,y).

        Override if it doesn't.
        :return: Array of training metrics.
        """
        pass

    def evaluate(self):
        """
        This method assumes your model instance uses sklearn style interface: ie. score(X,y).

        Override if it doesn't.
        :return: Array of metrics.
        """
        pass




