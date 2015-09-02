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
    self.test_data_props = {'source_path': './data/frames/custom_1.csv'}
    self.test_model_props = {'name': 'testClassifier'}
    self.test_model_params = {'C': 0.1, 'multi_class': 'ovr', 'solver': 'liblinear', 'penalty': 'l2', 'n_components': 5}

def run(exp):
    exp.display()
    exp.run()
    exp.report()


class customModel(Model):
    def init_model(self):
        C = self.parameters['C']
        n_components = self.parameters['n_components']
        mclass = self.parameters['multi_class']
        solver = self.parameters['solver']
        logistic = LogisticRegression()
        pca = RandomizedPCA(n_components=n_components)
        self.model = Pipeline(steps=[('pca', pca), ('logistic', logistic)])


class customRegressionModel(Model):
    def init_model(self):
        C = self.parameters['C']
        n_components = self.parameters['n_components']
        mclass = self.parameters['multi_class']
        solver = self.parameters['solver']
        regressor = LinearRegression()
        pca = RandomizedPCA(n_components=n_components)
        self.model = Pipeline(steps=[('pca', pca), ('logistic', regressor)])



class customData(Dataset):
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
        model = customModel(self.test_model_props, self.test_model_params)
        exp = Experiment(self.config_path, model=model)
        run(exp)

    def test_custom_data(self):
        data = customData(self.test_data_props)
        model = customModel(self.test_model_props, self.test_model_params)
        exp = Experiment(self.config_path, model=model, dataset=data)
        run(exp)


class TestCustomRegression(unittest.TestCase):
    def setUp(self):
        setup(self)
        self.config_path = './data/cfg/test_custom_regression.cfg'
        self.test_model_props['type'] = 'regression'

    def test_custom_model(self):
        model = customRegressionModel(self.test_model_props, self.test_model_params)
        exp = Experiment(self.config_path, model=model)
        run(exp)
