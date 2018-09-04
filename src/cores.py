#!/usr/bin/env python3

# Import
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import torch.nn.functional as F
from random import sample
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import KernelPCA


# Default paths
_d_projects_ = '../projects/'
_d_test_= '../test/'

# Classes
class Data:

    def __init__(self, p_data=''):

        self.p_data = p_data
        self.variable_types = []
        self.df = pd.DataFrame()
        self.features = None
        self.n_features = 0
        self.id_col = None
        self.ids = []
        self.env_cols = None
        self.envs = None
        self.env_coder = MultiLabelBinarizer()
        self.data_cols = None

    def load_data(self, p_data=''):

        if p_data:
            self.p_data = p_data

        if self.p_data:
            vtypes = pd.read_csv(self.p_data, nrows=1)
            self.df = pd.read_csv(self.p_data, skiprows=2, header=None)
            self.df.columns = vtypes.columns
            self.variable_types = vtypes.values[0]
            self._set_id_col()
            self._set_env_cols()
            self._set_data_cols()

        return self

    def featurize(self):

        if self.df.empty:
            self.load_data()

        self.features = None

        if self.envs is not None:
            # Has envs
            features = np.zeros([len(self.ids), len(self.envs), len(self.data_cols)])

            for r,_ in enumerate(self.df.iterrows()):
                data_id = self.ids.index(self.df.iloc[r][self.id_col].values[0])
                env_id = self._get_env_idx(self.env_coder.transform([self.df.iloc[r][self.env_cols].values]))

                features[data_id][env_id] = self.df.iloc[r][self.data_cols]

            self.features = np.concatenate(np.transpose(features, (1,0,2)), axis=1)
            
        else:
            # No envs
            features = []
            for data_id in self.ids:
                features.append(self.df.loc[data_id][self.data_cols])

            self.features = np.array(features)

        self.n_features = len(self.features[0])

        return

    def decompose(self):

        if self.features is None:
            self.featurize()

        decomposer = KernelPCA(remove_zero_eig=False)

        data_decomp = decomposer.fit_transform(self.features)

        print('First 3: %.3f' % (np.sum(decomposer.lambdas_[:1]) / np.sum(decomposer.lambdas_)))

        return data_decomp

    def visualize(self, env_var=0):

        sns.set(style='ticks')

        env_classes = self.df[self.env_cols[env_var]].unique()

        cols = []
        for c in self.data_cols:
            is_unique = True
            for v in env_classes:
                if len(self.df[self.df[self.env_cols[env_var]] == v][c].unique()) < 2:
                    is_unique = False
                    break

            if is_unique:
                cols.append(c)
            else:
                print('Ignore column "%s" due to lack of variation' % c)

        sns.pairplot(self.df, vars=cols, kind='reg', hue=self.env_cols[env_var])

        plt.show()

        return

    def _set_id_col(self):

        id_col = self.variable_types == 'id'

        if not id_col.any():

            raise ValueError('Must have an "id" column.')

        if np.sum(id_col) > 1:

            raise ValueError('Cannot have more than one "id" column.')

        self.id_col = self.df.columns[id_col]
        self.ids = list(self.df[self.id_col[0]].unique())

        return

    def _set_env_cols(self):

        env_cols = self.variable_types == 'env'

        self.env_cols = self.df.columns[env_cols]
        envs = self.env_coder.fit_transform(self.df[self.env_cols].values)
        self.envs = np.unique(envs, axis=0)

        return

    def _set_data_cols(self):

        data_cols = 1 * (self.variable_types != 'id') * (self.variable_types != 'env') == 1

        self.data_cols = self.df.columns[data_cols]

        return

    def _get_env_idx(self, env_code):

        return np.where((self.envs == env_code).all(axis=1))[0][0]

class Project:

    def __init__(self, p_project='', p_train_data='', p_eval_data='', p_predict_data=''):

        self.p_project = p_project

        self.train_data = Data(p_train_data).load_data()
        self.eval_data = Data(p_eval_data).load_data()
        self.predict_data = Data(p_predict_data).load_data()

    def save_project(self, p_project=''):

        if p_project:
            self.p_project = p_project
        else:
            self.p_project = _d_projects_ + 'glados_%s_%s.pkl' % (datetime.now().strftime('%Y%m%d'), ''.join(sample('abcdefgh12345678', 6)))

        pickle.dump(self, open(self.p_project, 'wb'))

        return self.p_project

    def load_project(self, p_project=''):

        if p_project:
            self.p_project = p_project

        proj = pickle.load(open(self.p_project, 'rb'))
        if isinstance(proj, Project):
            self.p_project = proj.p_project
            self.train_data = proj.train_data
            self.eval_data = proj.eval_data
            self.predict_data = proj.predict_data
        else:
            raise ValueError('Not a Glados Project class object.')

        return

    def assess_data(self):

        return

class NNModel(torch.nn.Module):

    def __init__(self, data=Data()):
        super(NNModel, self).__init__()

        self.data = data

        # layers
        self.fc1 = torch.nn.Linear(self.data.n_features, 2 * self.data.n_features)
        self.fc2 = torch.nn.Linear(2 * self.data.n_features, 4 * self.data.n_features)
        self.fc3 = torch.nn.Linear(4 * self.data.n_features, self.data.n_features)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Run
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Just keep on trying until you run out of cakes')

    parser.add_argument('-n', dest='path_new_project', action='store', nargs='?', default=-1, type=open, help='New project')
    parser.add_argument('-l', dest='path_project', action='store', nargs=1, default='', help='Load a project')
    parser.add_argument('-r', dest='run', help='try')
    parser.add_argument('--test', dest='test', action='store_true', help='Test build')

    args = parser.parse_args()

    if args.test:

        d = Data().load_data('../test/test.csv')
        x = d.decompose()
        d.visualize()
