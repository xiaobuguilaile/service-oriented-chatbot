# -*-coding:utf-8 -*-

'''
@File       : setup.py
@Author     : HW Shen
@Date       : 2020/5/20
@Desc       :
'''

from __future__ import print_function

import sys
from setuptools import setup, find_packages

from ServiceOrientedChatbot import __version__

# check python version
if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for dialog-bot')

with open("READ.md", "r", encoding="utf-8") as f:
    readme = f.read()

with open("LICENSE", "r", encoding="utf-8") as f:
    license = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    reqs = f.read()
''
