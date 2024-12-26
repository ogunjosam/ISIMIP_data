# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:33:37 2024

@author: ogunjosam
"""

import pandas as pd
import numpy as np
from itertools import groupby
import prcp
import glob


fl = glob.glob('*.csv')

for iv in fl:
    print(iv)
    xx = pd.read_csv(iv,index_col=[0])
    xx.index = pd.DatetimeIndex(xx.index)
    xx = xx[~((xx.index.month == 2) & (xx.index.day == 29))]

    loc = list(xx)

    xx['Year'] = xx.index.year
    xx['Month'] = xx.index.month

    hh = []
    for i in xx.Year.unique():
        
        xy = xx[xx['Year']==i]
        for k in loc:
            bb = xy[k]
            print([i,k,
                   prcp.rx1_day(bb.values)[0],
                   prcp.calculate_cwd(bb.values,thresh=1)[0],
                   prcp.calculate_cdd(bb.values,thresh=1)[0],
                   prcp.sdii(bb.values,threshold=1)[0],
                   prcp.r10mm(bb.values,threshold=1)[0]])
            hh.append([i,k,
                   prcp.rx1_day(bb.values)[0],
                   prcp.calculate_cwd(bb.values,thresh=1)[0],
                   prcp.calculate_cdd(bb.values,thresh=1)[0],
                   prcp.sdii(bb.values,threshold=1)[0],
                   prcp.r10mm(bb.values,threshold=1)[0]])

    hh_out = pd.DataFrame(hh,columns=['Year','Location','RX1DAY',
                                      'CWD','CDD','SDII','r1mm'])
    hh_out.to_excel(str.split(iv,'.')[0] +'_extremes.xlsx')