# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Script
"""

import numpy as np
from package.module import Function, Class

variable = 3
variable **= 2

output = Function(variable)

print(output)

def MyFunction(local_variable):
    local_variable += np.pi
    return local_variable

new_output = MyFunction(output)

results = [output, new_output]

np.save('results.npy', np.array(results))

