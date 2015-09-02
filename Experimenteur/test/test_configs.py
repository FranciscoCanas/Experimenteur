import unittest
import ConfigParser
from Experimenteur.dataset import Dataset

from Experimenteur.experiment import Experiment
from Experimenteur.model import Model

def setup(self):
    self.config_path = './data/cfg/test1.cfg'
    self.test_model_props = {'name': 'testClassifier', 'class': 'sklearn.ensemble.AdaBoostClassifier'}
    self.test_model_params = {'n_estimators': 5}
    self.test_data_props = {'source_path': './data/matrices/test_data.npy'}


def run(exp):
    exp.display()
    exp.run()
    exp.report()


class TestConfigs(unittest.TestCase):
    """
    Test config file readin'
    """
    def setUp(self):
        setup(self)

    def test_readConfig(self):
        config = ConfigParser.ConfigParser()
        config.read(self.config_path)
        for section in config.sections():
            print section
            print config.items(section)

    def test_initExperiment(self):
        exp = Experiment(self.config_path)

    def test_initExperimentCustomModel(self):
        model = Model(properties=self.test_model_props, parameters=self.test_model_params)

    def test_init_ExperimentCustomData(self):
        data = Dataset(self.test_data_props)


class TestExperiment(unittest.TestCase):
    def setUp(self):
        setup(self)


    def test_classify(self):
        exp = Experiment(self.config_path)
        run(exp)


class TestSplitDataExperiment(unittest.TestCase):
    def setUp(self):
        setup(self)
        self.config_path = './data/cfg/test_split_2.cfg'


    def test_classify(self):
        exp = Experiment(self.config_path)
        run(exp)


class TestRegressionExperiment(unittest.TestCase):
    def setUp(self):
        setup(self)
        self.config_path = './data/cfg/test_regression.cfg'


    def test_classify(self):
        exp = Experiment(self.config_path)
        run(exp)