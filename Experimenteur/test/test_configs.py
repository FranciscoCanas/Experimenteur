import unittest
import ConfigParser
from Experimenteur.dataset import Dataset

from Experimenteur.experiment import Experiment
from Experimenteur.model import Model


class TestConfigs(unittest.TestCase):
    """
    Test config file readin'
    """
    def setUp(self):
        self.config_path = './data/test1.cfg'
        self.test_model_props = {'name': 'testClassifier', 'class': 'sklearn.ensemble.AdaBoostClassifier'}
        self.test_model_params = {'n_estimators': 5}
        self.test_data_props = {'source_path': './data/test.npy'}

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
