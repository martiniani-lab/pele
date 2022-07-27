"""
Created on Jul 27, 2012

@author: vr274
"""
from __future__ import print_function

import pickle

from pele.storage import savenlowest
from pele.utils import dmagmin

save = savenlowest.SaveN(nsave=1000, accuracy=1e-3, compareMinima=dmagmin.compareMinima)

import sys

for i in sys.argv[1:]:
    print(i)
    save2 = pickle.load(open(i, "r"))
    for m in save2.data:
        save.insert(m.E, m.coords)

pickle.dump(save, open("storage", "w"))
