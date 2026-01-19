#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : setup.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/24/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import sys

from os.path import abspath, dirname

PYBULLET_CFG_DIR = dirname(dirname(dirname(dirname(__file__))))
sys.path.insert(0, abspath(PYBULLET_CFG_DIR))

from pybullet_engine.ikfast.compile import compile_ikfast
sys.argv[:] = sys.argv[:1] + ['build']


def main():
    robot_name = 'panda_arm'
    compile_ikfast(
        module_name='ikfast_{}'.format(robot_name),
        cpp_filename='ikfast_{}.cpp'.format(robot_name),
        remove_build=True
    )


if __name__ == '__main__':
    main()
