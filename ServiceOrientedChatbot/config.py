# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0":仅调用一次GPU，"1.0": 如果持续性的调用GPU

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Knowledge-Graph
# host = "127.0.0.1"
# kg_port = 7474
# user = "neo4j"
# password = "123456"
# answer_num_limit = 20

# Mongodb
mongo_host = 'localhost'
mongo_port = 27017

# Tokenize-config
punctuations_path = os.path.join(pwd_path, "data/punctuations.txt")
stopwords_path = os.path.join(pwd_path, "data/stopwords.txt")
user_define_words_path = os.path.join(pwd_path, "data/user_define_words.txt")
remove_words_path = os.path.join(pwd_path, "data/remove_words.txt")

# Log
log_file = os.path.join(pwd_path, 'output/log.txt')

# Search-dialog
local_model = 'bm25'
question_answer_path = os.path.join(pwd_path, 'data/chat/question_answer.tsv')
context_response_path = os.path.join(pwd_path, 'data/taobao/context_response.txt')
search_vocab_path =  os.path.join(pwd_path, 'data/chat/vocab.txt')

# Seq2seq-dialog
dialog_mode = 'single'
vocab_path = os.path.join(pwd_path, "data/taobao/vocab.txt")
model_path = os.path.join(pwd_path, 'output/')
seq2seq_model_path = os.path.join(model_path, dialog_mode)
predict_result_path = os.path.join(pwd_path, 'output/predict_result.txt')


class Params:
    """
    Seq2seq: autodecoder + attention model
    """

    layer_size = 256  # typical options : 128, 256, 512, 1024
    # embedding_size = 300
    embedding_size = 128
    # vocab_size = 10000
    encoder_vocab_size = 20000
    decoder_vocab_size = 20000
    learning_rate = 0.001
    batch_size = 128
    epochs = 15
    save_steps = 300
    model_name = "chatbot.ckpt"
    beam_size = 10
    max_gradient_norm = 5.0
    use_attention = True
    bidirectional_rnn = False

    # train data
    train_data = os.path.join(pwd_path +'data/train.data')
    resource_data = os.path.join(pwd_path +'data/xiaohuangji50w_nofenci.conv')  # original data
    model_data = os.path.join(pwd_path +'data/model_data')

    max_train_data_size = 50000  # normally no limit

class Web:
    """
    前端
    """
    host = "0.0.0.0"
    port = "8820"
    url = "http://" + host + ":" + port
