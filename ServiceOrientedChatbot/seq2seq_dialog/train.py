# -*-coding:utf-8 -*-

'''
@File       : train.py
@Author     : HW Shen
@Date       : 2020/5/26
@Desc       : 训练和预测代码
'''

import os
import sys
import time
import tensorflow as tf
import io

from ServiceOrientedChatbot import config
from ServiceOrientedChatbot.seq2seq_dialog import seq2seq

PARAMS = config.Params

# seq2seq model PARAMS
vocab_input_size = PARAMS.encoder_vocab_size
vocab_target_size = PARAMS.decoder_vocab_size
embedding_size = PARAMS.embedding_size
layer_size = PARAMS.layer_size
batch_size = PARAMS.batch_size

# define max length
max_input_length, max_target_length = 20, 20


def max_length(tensor):

    return max([len(t) for t in tensor])


def preprocess_sentence(s):
    """
    每个句子首尾加上标识符
    s: " "连接的分词
    """
    s = 'start' + s + 'end'
    return s


def tokenize(text):

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_input_size, oov_token=3)
    lang_tokenizer.fit_on_texts(text)

    vector = lang_tokenizer.texts_to_sequences(text)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        vector, maxlen=max_input_length, padding='post')

    return tensor, lang_tokenizer


def create_dataset(data_path, num_examples):

    lines = io.open(data_path, encoding='utf-8').read().strip().split('\n')
    # [[x:y],[],[]]
    word_pairs = [[preprocess_sentence(sentence) for sentence in line.split('\t')] for line in lines[:num_examples]]

    return zip(*word_pairs)


def read_data(data_path, num_examples):

    input_lang, target_lang = create_dataset(data_path, num_examples)

    input_tensor, input_tokenizer = tokenize(input_lang)
    target_tensor, target_tokenizer = tokenize(target_lang)

    return input_tensor, input_tokenizer, target_tensor, target_tokenizer

# 读取数据
x_input_tensor, input_tokenizer, y_target_tensor, target_tokenizer = read_data(PARAMS.train_data, PARAMS.max_train_data_size)


def train():
    """ 训练模型 """

    print(" --- preparing data --- ")

    steps_per_epoch = len(x_input_tensor) // PARAMS.batch_size
    print("steps_per_epoch: ", steps_per_epoch)

    encoder_hidden = seq2seq.encoder.initialize_hidden_state()
    checkpoint_dir = PARAMS.model_data

    # 取出 checkpoint_dir 中对应的模型数据
    ckpt = tf.io.gfile.listdir(checkpoint_dir)

    if ckpt:
        print("reload pretrained model")
        seq2seq.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))  # 恢复最后一次训练的模型数据

    batch_size = len(x_input_tensor)

    dataset = tf.data.Dataset.from_tensor_slices((x_input_tensor, y_target_tensor)).shuffle(batch_size)  # 随机打乱训练数据

    start_time = time.time()

    # 模型数据存入新路径
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    current_steps = 0

    while True:
        start_time_epoch = time.time()
        total_loss = 0
        for (batch, (x_input, y_target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = seq2seq.batch_train_step(x_input, y_target, target_tokenizer, encoder_hidden)
            total_loss += batch_loss
            print("Currnet Batch is: {}, Current loss is : {}".format(batch, batch_loss.numpy()))  # 当前损失

        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps += steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps

        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(
            current_steps, step_time_total, step_time_epoch, step_loss.numpy()))

        seq2seq.checkpoint.save(file_prefix=checkpoint_prefix)

        sys.stdout.flush()


def predict(sentence):
    """
    预测
    sentence: " "连接的分词
    """
    checkpoint_dir = PARAMS.model_data
    seq2seq.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    sentence = preprocess_sentence(sentence)
    x_inputs = [input_tokenizer.word_index.get(word, 3) for word in sentence.split(' ')]
    padding_x = tf.keras.preprocessing.sequence.pad_sequences([x_inputs], max_input_length, padding='post')  # 末尾+0的padding

    result = ''
    hidden = [tf.zeros((1, PARAMS.layer_size))]
    encoder_output, encoder_hidden = seq2seq.encoder(padding_x, hidden)

    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([target_tokenizer.word_index['start']], 0)

    for t in range(max_target_length):
        y_preds, decoder_hidden, attention_weights = seq2seq.decoder(decoder_input, decoder_hidden, encoder_output)

        predicted_id = tf.argmax(y_preds[0]).numpy()  # 找出预测最大概率的结果编号

        if target_tokenizer.index_word[predicted_id] == 'end':
            break

        result += target_tokenizer.index_word[predicted_id] + ' '

        decoder_input = tf.expand_dims([predicted_id], 0)

    return result


if __name__ == '__main__':
    
    s = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    print(zip(*s))
    print(zip(*s))
    print(list(zip(*s)))