# -*-coding:utf-8 -*-

'''
@File       : model.py
@Author     : HW Shen
@Date       : 2020/5/27
@Desc       :
'''
import io
import os
import sys
import time

import tensorflow as tf
from ServiceOrientedChatbot import config


class Encoder(tf.keras.Model):
    """

    """
    def __init__(self, vocab_size, embedding_size, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru = tf.keras.layers.GRU(
            self.encoder_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, padding_x, hidden):
        padding_x = self.embedding(padding_x)
        encoder_output, encoder_hidden = self.gru(padding_x, initial_state=hidden)

        return encoder_output, encoder_hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))


class BahdanauAttention(tf.keras.Model):
    """
    注意力层
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):

        # hidden_with_time_axis shape: (batch_size, 1, hidden_size)， hidden shape: (batch_size, hidden size)
        hidden_with_time_axis = tf.expand_dims(query, 1)  # perform addition to calculate the score

        # score shape: (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape: (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # get 1 at the last axis because we apply score to self.V

        # context_vector shape: (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """

    """
    def __init__(self, vocav_size, embedding_size, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(vocav_size, embedding_size)
        self.gru = tf.keras.layers.GRU(self.decoder_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocav_size)
        self.attention = BahdanauAttention(self.decoder_units)

    def call(self, x, hidden, encoder_units):

        context_vector, attention_weights = self.attention(hidden, encoder_units)  # 获取 上下文向量 和 注意力权重

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=1)
        decoer_output, decoder_hidden = self.gru(x)

        output = tf.reshape(decoer_output, (-1, decoer_output.shape[2]))

        y_preds = self.fc(output)

        return y_preds, decoder_hidden, attention_weights


# seq2seq model PARAMS
PARAMS = config.Params
vocab_input_size = PARAMS.encoder_vocab_size
vocab_target_size = PARAMS.decoder_vocab_size
embedding_size = PARAMS.embedding_size
layer_size = PARAMS.layer_size
batch_size = PARAMS.batch_size

# encoder layer + decoder layer
encoder = Encoder(vocab_input_size, embedding_size, layer_size, batch_size)
decoder = Decoder(vocab_target_size, embedding_size, layer_size, batch_size)
optimizer = tf.keras.optimizers.Adam()

# define max length
max_input_length, max_target_length = 20, 20

checkpoint = tf.train.Checkpoint(
    optimizer=optimizer,
    encoder=encoder,
    decoder=decoder
)


class Seq2SeqModel(object):

    def __init__(self):
        # 读取数据
        self.x_input_tensor, self.input_tokenizer, self.y_target_tensor, self.target_tokenizer = self.read_data(PARAMS.train_data,
                                                                                       PARAMS.max_train_data_size)

    def loss_function(self, y, y_preds):
        """
        损失函数
        y: 真实值
        y_pred: 预测值
        """
        mask = tf.math.logical_not(tf.math.equal(y, 0))

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_ = loss_object(y, y_preds)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def batch_train_step(self, x_input, y_target, y_tar_tokenizer, encoder_hidden):
        """
        input: x
        target:
        target_lang:
        encoder_hidden:
        """

        loss = 0
        with tf.GradientTape() as tape:

            # encoder自动调用Encoder()的call
            encoder_output, encoder_hidden = encoder(x_input, encoder_hidden)
            decoder_hidden = encoder_hidden

            decoder_input = tf.expand_dims([y_tar_tokenizer.word_index['start']] * batch_size, 1)

            for t in range(1, y_target.shape[1]):

                y_preds, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)

                loss += self.loss_function(y_target[:, t], y_preds)  # 累加 loss 求和

                decoder_input = tf.expand_dims(y_target[:, t], 1)

        batch_avg_loss = (loss / int(y_target.shape[1]))  # 计算 batch 的平均损失

        variables = encoder.trainable_variables + decoder.trainable_variables  # 整合所有变量
        gradients = tape.gradient(loss, variables)  # 计算各变量梯度
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_avg_loss

    def train(self):
        """ 训练模型 """

        print(" --- preparing data --- ")

        steps_per_epoch = len(self.x_input_tensor) // PARAMS.batch_size
        print("steps_per_epoch: ", steps_per_epoch)

        encoder_hidden = encoder.initialize_hidden_state()
        checkpoint_dir = PARAMS.model_data

        # 取出 checkpoint_dir 中对应的模型数据
        ckpt = tf.io.gfile.listdir(checkpoint_dir)

        if ckpt:
            print("reload pretrained model")
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))  # 恢复最后一次训练的模型数据

        batch_size = len(self.x_input_tensor)

        dataset = tf.data.Dataset.from_tensor_slices((self.x_input_tensor, self.y_target_tensor)).shuffle(batch_size)  # 随机打乱训练数据

        start_time = time.time()

        # 模型数据存入新路径
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

        current_steps = 0

        while True:
            start_time_epoch = time.time()
            total_loss = 0
            for (batch, (x_input, y_target)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.batch_train_step(x_input, y_target, self.target_tokenizer, encoder_hidden)
                total_loss += batch_loss
                print("Currnet Batch is: {}, Current loss is : {}".format(batch, batch_loss.numpy()))  # 当前损失

            step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
            step_loss = total_loss / steps_per_epoch
            current_steps += steps_per_epoch
            step_time_total = (time.time() - start_time) / current_steps

            print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(
                current_steps, step_time_total, step_time_epoch, step_loss.numpy()))

            checkpoint.save(file_prefix=checkpoint_prefix)

            sys.stdout.flush()

    def predict(self, sentence):
        """
        预测
        sentence: " "连接的分词
        """
        checkpoint_dir = PARAMS.model_data
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        sentence = self.preprocess_sentence(sentence)
        x_inputs = [self.input_tokenizer.word_index.get(word, 3) for word in sentence.split(' ')]
        padding_x = tf.keras.preprocessing.sequence.pad_sequences([x_inputs], max_input_length,
                                                                  padding='post')  # 末尾+0的padding

        result = ''
        hidden = [tf.zeros((1, PARAMS.layer_size))]
        encoder_output, encoder_hidden = encoder(padding_x, hidden)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.target_tokenizer.word_index['start']], 0)

        for t in range(max_target_length):
            y_preds, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_output)

            predicted_id = tf.argmax(y_preds[0]).numpy()  # 找出预测最大概率的结果编号

            if self.target_tokenizer.index_word[predicted_id] == 'end':
                break

            result += self.target_tokenizer.index_word[predicted_id] + ' '

            decoder_input = tf.expand_dims([predicted_id], 0)

        return result

    def max_length(self, tensor):

        return max([len(t) for t in tensor])

    def preprocess_sentence(self, s):
        """
        每个句子首尾加上标识符
        s: " "连接的分词
        """
        s = 'start' + s + 'end'
        return s

    def tokenize(self, text):

        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_input_size, oov_token=3)
        lang_tokenizer.fit_on_texts(text)

        vector = lang_tokenizer.texts_to_sequences(text)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(
            vector, maxlen=max_input_length, padding='post')

        return tensor, lang_tokenizer

    def create_dataset(self, data_path, num_examples):

        lines = io.open(data_path, encoding='utf-8').read().strip().split('\n')
        # [[x:y],[],[]...]
        word_pairs = [[self.preprocess_sentence(sentence) for sentence in line.split('\t')] for line in lines[:num_examples]]

        return zip(*word_pairs)

    def read_data(self, data_path, num_examples):

        input_lang, target_lang = self.create_dataset(data_path, num_examples)

        input_tensor, input_tokenizer = self.tokenize(input_lang)
        target_tensor, target_tokenizer = self.tokenize(target_lang)

        return input_tensor, input_tokenizer, target_tensor, target_tokenizer

