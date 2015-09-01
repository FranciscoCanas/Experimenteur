class Model:
    """
    Represents a statistical model. Must implement a function used to fit/train, and a function
    used to evaluate.
    """

    def __init__(self, configuration):
        self.properties = dict(configuration.items('model'))
        self.parameters = dict(configuration.items('parameters'))
        self.name = self.properties.get('name', 'model_name')
        self.init_model()


    def init_model(self):
        """
        Instantiate the model:
        ie. skl
        """
        model_class = self.parameters.get('model_class')
        split_name = model_class.split('.')
        class_name = split_name[-1]
        module_name = '.'.join(split_name[:-1])
        module = __import__(module_name)
        class_ = getattr(module, class_name)
        self.model = class_(**self.parameters)


    def fit(self):
        pass

    def evaluate(self):
        pass




