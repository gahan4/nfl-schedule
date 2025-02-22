#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:08:48 2025

@author: neil
"""
import pandas as pd

NUM_TEAMS = int(32)
NUM_WEEKS = int(18)
NUM_MATCHUPS = int(32 * (18 - 1) / 2)

slots = pd.DataFrame({
    'slot_id': range(4),
    'slot_desc': ['Sun', 'TNF', 'SNF', 'MNF']})

NUM_SLOTS = slots.shape[0]
