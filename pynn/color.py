"""
Some utility functions used for printing colored text.

Yujia Li, 09/2014
"""

_GOOD_COLOR_BEGINS = '\033[42m'
_BAD_COLOR_BEGINS = '\033[41m'
_COLOR_RESET = '\033[0m'

def good_colored_str(txt):
    return _GOOD_COLOR_BEGINS + txt + _COLOR_RESET

def bad_colored_str(txt):
    return _BAD_COLOR_BEGINS + txt + _COLOR_RESET


