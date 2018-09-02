#!/usr/bin/env python3

# Import
import argparse
import numpy as np
import pandas as pd
from random import sample
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
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
        self.features = []
        self.id_col = None
        self.ids = []
        self.env_cols = None
        self.envs = {}
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

        self.features = []

        if self.env_cols is not None:
            feature0 = np.zeros(len(self.env_cols) * len(self.data_cols))
            
        else:
            feature0 = np.zeros(len(self.data_cols))

        return

    def decompose(self):

        if self.features.empty:
            self.featurize()

        decomposer = KernelPCA()

        data_decomp = decomposer.fit_transform(self.features.values)

        print('First 3: %.3f' % (np.sum(decomposer.lambdas_[:3]) / np.sum(decomposer.lambdas_)))

        return data_decomp

    def _set_id_col(self):

        id_col = self.variable_types == 'id'

        if np.sum(id_col) > 1:

            raise ValueError('Cannot have more than one "id" column.')

        self.id_col = self.df.columns[id_col]
        self.ids = self.df[self.id_col[0]].unique()

        return

    def _set_env_cols(self):

        env_cols = self.variable_types == 'env'

        self.env_cols = self.df.columns[env_cols]
        self.envs = {c:self.df[c].unique() for c in self.env_cols}

        return

    def _set_data_cols(self):

        data_cols = 1 * (self.variable_types != 'id') * (self.variable_types != 'env') == 1

        self.data_cols = self.df.columns[data_cols]

        return

class Project:

    def __init__(self, p_project='', p_train_data='', p_eval_data='', p_predict_data=''):

        self.p_project = p_project

        self.train_data = Data(p_train_data).load_data()
        self.eval_data = Data(p_eval_data).load_data()
        self.predict_data = Data(p_predict_data).load_data()

    def new_project(self, p_project=''):

        if p_project:

            self.p_project = p_project

        else:

            self.p_project = _d_projects_ + 'project_%s_%s.glados' % (datetime.now().strftime('%Y%m%d'), ''.join(sample('abcdefgh12345678', 6)))

        if self.p_project:

            with open(self.p_project, 'w+') as f:

                # TODO create new project file

                _ = f.write('')

        return self.p_project

    def load_project(self, p_project=''):

        if p_project:

            self.p_project = p_project

        # TODO read in project

        return

    def assess_data(self):

        """
        Let's first examine for:
            1. Leak
            2. Clusters

        :return:
        """

        return

# Run
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Just keep on trying until you run out of cakes')

    parser.add_argument('-n', dest='path_new_project', action='store', nargs='?', default=-1, type=open, help='New project')
    parser.add_argument('-l', dest='path_project', action='store', nargs=1, default='', help='Load a project')
    parser.add_argument('-i', dest='test', help='try')

    args = parser.parse_args()

    project = Project()

    if args.path_new_project != -1:
        project.new_project(args.path_new_project)

    if args.path_project:
        project.load_project(args.path_project)
