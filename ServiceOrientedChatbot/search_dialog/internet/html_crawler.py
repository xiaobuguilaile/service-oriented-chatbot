# -*-coding:utf-8 -*-

'''
@File       : html_crawler.py
@Author     : HW Shen
@Date       : 2020/5/25
@Desc       :
'''

import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'
}


def get_html_zhidao(url):
    """
    获取百度知道的页面
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_baike(url):
    """
    获取百度百科的页面
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_baidu(url):
    """
    获取百度搜索的结果
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_bingwd(url):
    """
    获取Bing网网典的页面
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_bing(url):
    """
    获取Bing网搜索的结果
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


