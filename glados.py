#!/usr/bin/env python3

# Import
import os
import argparse
from src.cores import Project
from src.iofunc import *

# Default paths
_d_glados_ = os.path.realpath(os.path.split(__file__)[0]) + '/'
_d_projects_ = _d_glados_ + 'projects/'
_d_test_= _d_glados_ + 'test/'

# Run
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Just keep on trying until you run out of cakes')

    parser.add_argument('-n', dest='path_new_project', action='store', nargs='?', default=-1, type=open, help='New project')
    parser.add_argument('-l', dest='path_project', action='store', nargs=1, default='', help='Load a project')
    parser.add_argument('-r', dest='run', help='try')
    parser.add_argument('--save', dest='save', action='store_true', help='Save project')
    parser.add_argument('--plot', dest='plot', action='store_true', help='Show plots')
    parser.add_argument('--test', dest='test', action='store_true', help='Test build')
    parser.add_argument('--interact', dest='interact', action='store_true', help='To interact with local vars')

    args = parser.parse_args()

    p = Project()

    if args.save:
        p_file = _d_projects_ + 'glados_' + gen_file_stamp()
        p.save_project(p_project=p_file)

    if args.test:

        p = Project(p_train_data='test/test.csv')
        p.analyze_data(plot=args.plot)

    if args.interact:
        interact(local=locals())
