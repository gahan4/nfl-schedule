#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:49:59 2025

@author: neil
"""
import pandas as pd
from create_constraints import get_index
import numpy as np

def create_objective_function(teams, A_in, A_eq):
    return np.zeros(A_in.shape[1])