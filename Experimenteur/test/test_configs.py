import unittest

from Experimenteur.experiment import Experiment

class TestConfigs(unittest.TestCase):
    """
    Test config file readin'
    """
    def setUp(self):
        self.config_path = './data/test1.cfg'

    def test_readConfig(self):
        exp = Experiment(self.config_path)
