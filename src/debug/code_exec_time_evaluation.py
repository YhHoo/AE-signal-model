'''
This code is for us to evaluate how fast our code finish execution
Just update the param setup to the code that run only once e.g. import lib, const initialize,
np_code, sp_code which represent the 2 codes that are to be compared in exec time.
'''
from timeit import timeit

# your only-run-once code
setup = '''
import numpy as np
from scipy.signal import correlate as correlate_scipy
from numpy import correlate as correlate_numpy
l = np.random.rand(10000)
m = np.random.rand(10000)
'''

# your code that is to be compared
np_code_title = None
sp_code_title = None
np_code = 'correlate_numpy(l, m, \'full\')'
sp_code = 'correlate_scipy(l, m, \'full\', method=\'fft\')'

# result displaying
print('-----------[Code execution speed test]-------------')
print('{} exec time: '.format(np_code_title), timeit(setup=setup, stmt=np_code, number=1))
print('{} exec time: '.format(sp_code_title), timeit(setup=setup, stmt=sp_code, number=1))
