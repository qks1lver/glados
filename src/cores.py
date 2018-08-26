#!/usr/bin/env python3

# Import
import argparse
import numpy as np
import pandas as pd
from random import sample
from datetime import datetime


# Default paths
_d_projects_ = '../projects/'
_d_test_= '../test/'

# Classes
class Data():

    def __init__(self, p_data=''):

        self.p_data = p_data
        self.variable_types = []
        self.df = pd.DataFrame()

    def load_data(self, p_data=''):

        if p_data:
            self.p_data = p_data

        if self.p_data:
            vtypes = pd.read_csv(self.p_data, nrows=2)
            self.df = pd.read_csv(self.p_data, skiprows=2, header=None)
            self.df.columns = vtypes.columns
            self.variable_types = np.array(vtypes.columns)

        return self

class Project():

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

    parser.add_argument('-n', dest='path_new_project', action='store', nargs='?', default=-1, type=open, help='Start a new project')
    parser.add_argument('-l', dest='path_project', action='store', nargs=1, default='', help='Load a project')
    parser.add_argument('-i', dest='test', help='try')

    args = parser.parse_args()

    project = Project()

    if args.path_new_project != -1:
        project.new_project(args.path_new_project)

    if args.path_project:
        project.load_project(args.path_project)
