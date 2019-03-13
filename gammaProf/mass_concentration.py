import numpy as np

def child2018(m200c, fit='nfw'):
    
    # Mass-concentration relation from Child+2018
    
    p_all = {'m':-0.10, 'A':3.44, 'b':430.49, 'c0':3.19}
    p_relax = {'m':-0.09, 'A':2.88, 'b':1644.53, 'c0':3.54}
    p_NFW = {'m':-0.07, 'A':4.61, 'b':638.65, 'c0':3.59}
    p_einasto = {'m':-0.01, 'A':63.2, 'b':431.48, 'c0':3.36}
    p = {'all':p_all, 'relax':p_relax, 'nfw':p_NFW, 'einasto':p_einasto}
    
    m = p[fit]['m']
    A = p[fit]['A']
    b = p[fit]['b']
    c0 = p[fit]['c0']
    mstar = 1e12
    
    return A * ((m200c/mstar/b)**m * (1 + m200c/mstar/b)**(-m) -1) + c0
