# -*-coding:utf-8 -*-

'''
@File       : search_engine.py
@Author     : HW Shen
@Date       : 2020/5/25
@Desc       : 优先搜索 baidu+bing, 如果仍然没有结果，尝试搜索剩余部分的结果
'''

from collections import OrderedDict, defaultdict
from urllib.parse import quote

from ServiceOrientedChatbot.search_dialog.internet import html_crawler
from ServiceOrientedChatbot.utils.logger import logger
from ServiceOrientedChatbot.utils.tokenizer import postag


bing_url_prefix = 'https://cn.bing.com/search?q='  # 微软bing搜索
baidu_url_prefix = 'https://www.baidu.com/s?ie=UTF-88&wd='  # 百度搜索
calendar_url = 'http://open.baidu.com/calendar'  # 百度日历
calculator_url = 'http://open.baidu.com/static/calculator/calculator.html'  # 百度计算器
weather_url = 'http://www.weather.com.cn'  # 天气

split_symbols = ["。", "?", ".", "_", "-", ":", "！", "？"]  # 常见标点


def split_2_short_text(sentence):
    """ 通过 '\t' 把sentence分割成short text """

    print(sentence)
    for symbol in split_symbols:
        sentence = sentence.replace(symbol, symbol+"\t")

    return sentence.split("\t")  # 把结尾的"\t"去掉


def keep_pos_words(query, tags=['n']):
    """ 只保留query中指定词性tags的词 """
    result = []
    words = postag(query)  # [(word, tag),(),()...]
    for k in words:
        if k.flag in tags:
            result.append(k.word)
    return result


class SearchEngine(object):

    """ 页面搜索引擎 """

    def __init__(self, topk=10):
        self.name = "engine"
        self.topk = topk
        self.contents = OrderedDict()

    def search(self, query):
        """
        通过baidu, bing等 返回对query的搜索结果
        """
        # 检索baidu
        print("---进入Baidu匹配---")
        r, baidu_left_text = self.search_baidu(query)
        if r:
            self.contents[query] = r
            return r

        # 检索bing
        print("---进入Bing匹配---")
        r, bing_left_text = self.search_bing(query)
        if r:
            self.contents[query] = r
            return r

        # 检索 baidu+bing 的摘要
        print("---进入其他匹配---")
        r = self._search_other(query, baidu_left_text + bing_left_text)
        if r:
            self.contents[query] = r
            return r
        return r

    def search_baidu(self, query):
        """
        通过baidu检索答案，包括百度的知识图谱、百度诗词、百度万年历、百度计算器、百度知道等
        """
        answer, left_text = [], ''

        # 抓取百度搜索的前10条摘要结果
        soup_baidu = html_crawler.get_html_baidu(baidu_url_prefix + quote(query))

        if not soup_baidu:
            return answer, left_text

        for i in range(1, self.topk):
            items = soup_baidu.find(id=1)  # 获取id=1的标签项(即第一个搜索结果)
            print("i= {}, items= {}".format(i, items))

            if not items:
                logger.debug("百度找不到答案")
                break

            # 判断是否有mu，如果第一个是百度知识图谱的，就直接命中答案
            # 百度知识图谱
            if ("mu" in items.attrs) and i == 1:
                r = items.find(class_='op_exactqa_s_answer')
                if r:
                    logger.debug("百度知识图谱中找到答案")
                    answer.append(r.get_text().strip())
                    return answer, left_text

            # 百度古诗词
            if ("mu" in items.attrs) and i == 1:
                r = items.find(class_='op_exactqa_detail_s_answer')
                if r:
                    logger.debug("百度知识图谱中找到答案")
                    answer.append(r.get_text().strip())
                    return answer, left_text

            # 百度万年历 & 日期
            if ('mu' in items.attrs) and i == 1 and items.attrs['mu'].__contains__(calendar_url):
                r = items.find(class_="op-calendar-content")
                if r:
                    logger.debug("百度万年历找到答案")
                    answer.append(r.get_text().strip().replace("\n", "").replace(" ", ""))
                    return answer, left_text
            if ('tpl' in items.attrs) and i == 1 and items.attrs['tpl'].__contains__('calendar_new'):
                r = items.attrs['fk'].replace("6018_", "")
                logger.debug(r)
                if r:
                    logger.debug("百度万年历新版找到答案")
                    answer.append(r)
                    return answer, left_text

            # 百度搜索的计算器
            if ('mu' in items.attrs) and i == 1 and items.attrs['mu'].__contains__(calculator_url):
                r = items.find(class_="op_new_val_screen_result")
                if r:
                    logger.debug("计算器找到答案")
                    answer.append(r.get_text().strip())
                    return answer, left_text

            # 百度搜索的天气
            if ('mu' in items.attrs) and i == 1 and items.attrs['mu'].__contains__(weather_url):
                r = items.find(class_="op_weather4_twoicon_today")
                if r:
                    logger.debug("天气找到答案")
                    answer.append(r.get_text().replace('\n', '').strip())
                    return answer, left_text

            # 百度知道
            if ('mu' in items.attrs) and i == 1:
                r = items.find(class_='op_best_answer_question_link')
                if r:
                    zhidao_soup = html_crawler.get_html_zhidao(r['href'])
                    r = zhidao_soup.find(class_='bd answer').find('pre')
                    if not r:
                        r = zhidao_soup.find(class_='bd answer').find(class_='line content').\
                            find(class_="best-text mb-10")
                    if r:
                        logger.debug("百度知道找到答案")
                        answer.append(r.get_text().strip().replace("展开全部", "").strip())
                        return answer, left_text
            # 百度知道的另一种形式
            if items.find("h3"):
                if items.find("h3").find("a").get_text().__contains__("百度知道") and (i == 1 or i == 2):
                    url = items.find("h3").find("a")['href']
                    if url:
                        zhidao_soup = html_crawler.get_html_zhidao(url)
                        r = zhidao_soup.find(class_='bd answer')
                        if r:
                            r = r.find('pre')
                            if not r:
                                r = zhidao_soup.find(class_='bd answer').find(class_='line content').find(
                                    class_="best-text mb-10")
                            if r:
                                logger.debug("百度知道找到答案")
                                answer.append(r.get_text().strip().replace("展开全部", "").strip())
                                return answer, left_text

                # 百度百科
                if items.find("h3").find("a").get_text().__contains__("百度百科") and (i == 1 or i == 2):
                    url = items.find("h3").find("a")['href']
                    if url:
                        logger.debug("百度百科找到答案")
                        baike_soup = html_crawler.get_html_baike(url)

                        r = baike_soup.find(class_='lemma-summary')
                        if r:
                            answer.append(r.get_text().replace("\n", "").strip())
                            return answer, left_text

            # 如果上面的全部匹配失败，就加入到left_text中
            left_text += items.get_text()

        print("baidu: ", left_text)
        return answer, left_text

    @staticmethod
    def search_bing(query):
        """
        通过微软bing检索答案，包括bing知识图谱，bing网典
        """
        answer, left_text = [], ''

        # 获取bing搜索结果的摘要
        soup_bing = html_crawler.get_html_bing(bing_url_prefix + quote(query))

        # 判断是否在bing的知识图谱中
        r = soup_bing.find(class_="bm_box")

        if r:
            r = r.find_all(class_="b_vList")
            if r and len(r) > 1:
                r = r[1].find("li").get_text().strip()
                if r:
                    answer.append(r)
                    logger.debug("Bing知识图谱找到答案")
                    return answer, left_text

        # Bing网典中查找
        else:
            r = soup_bing.find(id="b_results")
            if r:
                bing_list = r.find_all('li')
                for bl in bing_list:
                    temp = bl.get_text()
                    if temp.__contains__(" - 必应网典"):
                        logger.debug("查找Bing网典")
                        # Bing 网典链接
                        url = bl.find("h2").find("a")['href']
                        if url:
                            bingwd_soup = html_crawler.get_html_bingwd(url)
                            r = bingwd_soup.find(class_='bk_card_desc').find("p")
                            if r:
                                r = r.get_text().replace("\n", "").strip()
                                if r:
                                    logger.debug("Bing网典找到答案")
                                    answer.append(r)
                                    return answer, left_text
        left_text += r.get_text()

        print("bing: ", left_text)
        return answer, left_text

    def _search_other(self, query, left_text):
        """
        如果 baidu + bing 知识图谱中都没找到答案，那么就分析摘要
        """

        answer = []
        keywords = keep_pos_words(query)  # 取query中的名词n为核心词
        sentences = split_2_short_text(left_text.strip())  # 对left_text进行分句

        # 找出包含核心词的分句 key_sentences，其他的丢弃
        key_sentences = set()
        for s in sentences:
            for k in keywords:
                if k in s:
                    key_sentences.add(s)

        # 从核心句key_sentences中提取人名
        key_persons = self.key_items_by_pos(key_sentences)

        # 候选人列表, 需要剔除人名中的keywords
        candidate_persons = []
        for i, v in enumerate(key_persons):
            if v[0] not in keywords:
                candidate_persons.append(v)

        if candidate_persons:
            answer.extend(candidate_persons[:3])  # 添加排名前3个person

        return answer

    @staticmethod
    def key_items_by_pos(sentences, pos='nr'):  # 'nr' 表示人名
        target_dict = {}
        for ks in sentences:
            words = postag(ks)
            for w in words:
                if w.flag == pos:
                    if w.word in target_dict:
                        target_dict[w.word] += 1
                    else:
                        target_dict[w.word] = 1
        # 将人名按从大到小词频排序
        key_persons = sorted(target_dict.items(), key=lambda item:item[1], reverse=True)

        return key_persons


if __name__ == '__main__':

    # dic = {"zhangsan":5,
    #        "wangwu":2,
    #        "zhaosi":6}
    # key_persons = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    #
    # print(key_persons)

    engine = SearchEngine()
    query = "我今天在中山公园碰到了前阿里巴巴董事长马云"
    # query = "中山公园"
    r = engine.search(query)

    print(r)