import unittest
import ConfigParser
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from Experimenteur.dataset import Dataset

from Experimenteur.experiment import Experiment
from Experimenteur.model import Model


def setup(self):
    self.config_path = './data/cfg/test_custom.cfg'

def run(exp):
    exp.display()
    exp.run()
    exp.report()


class CustomModel(Model):
    def init_model(self):
        C = self.parameters['C']
        n_components = self.parameters['n_components']
        mclass = self.parameters['multi_class']
        solver = self.parameters['solver']
        logistic = LogisticRegression()
        pca = RandomizedPCA(n_components=n_components)
        self.model = Pipeline(steps=[('pca', pca), ('logistic', logistic)])


class CustomRegressionModel(Model):
    def init_model(self):
        C = self.parameters['C']
        n_components = self.parameters['n_components']
        mclass = self.parameters['multi_class']
        solver = self.parameters['solver']
        regressor = LinearRegression()
        pca = RandomizedPCA(n_components=n_components)
        self.model = Pipeline(steps=[('pca', pca), ('logistic', regressor)])



class CustomData(Dataset):
    def load_data(self, fold=''):
        d = pd.read_csv(self.properties['source_path'])
        self.X = d.ix[:,'V_0':].as_matrix()
        self.y = d['class'].as_matrix()
        print self.X.shape
        print self.y.shape
        return



class TestCustom(unittest.TestCase):
    def setUp(self):
        setup(self)

    def test_custom_model(self):
        exp = Experiment(self.config_path, model=CustomModel)
        run(exp)

    def test_custom_data(self):
        config_path = './data/cfg/test_custom_dataset.cfg'
        exp = Experiment(config_path, model=CustomModel, dataset=CustomData)
        run(exp)


class TestCustomRegression(unittest.TestCase):
    def setUp(self):
        setup(self)
        self.config_path = './data/cfg/test_custom_regression.cfg'

    def test_custom_model(self):
        exp = Experiment(self.config_path, model=CustomRegressionModel)
        run(exp)
