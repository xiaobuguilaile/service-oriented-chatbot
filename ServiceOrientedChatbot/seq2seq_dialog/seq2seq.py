# -*-coding:utf-8 -*-

'''
@File       : seq2seq.py
@Author     : HW Shen
@Date       : 2020/5/26
@Desc       :
'''


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


PARAMS = config.Params

# seq2seq model PARAMS
vocab_input_size = PARAMS.encoder_vocab_size
vocab_target_size = PARAMS.decoder_vocab_size
embedding_size = PARAMS.embedding_size
layer_size = PARAMS.layer_size
batch_size = PARAMS.batch_size

# encoder layer + decoder layer
encoder = Encoder(vocab_input_size, embedding_size, layer_size, batch_size)
decoder = Decoder(vocab_target_size, embedding_size, layer_size, batch_size)
optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(
    optimizer=optimizer,
    encoder=encoder,
    decoder=decoder
)


def loss_function(y, y_preds):
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


def batch_train_step(x_input, y_target, target_lang, encoder_hidden):
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

        decoder_input = tf.expand_dims([target_lang.word_index['start']] * batch_size, 1)

        for t in range(1, y_target.shape[1]):

            y_preds, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)

            loss += loss_function(y_target[:, t], y_preds)  # 累加 loss 求和

            decoder_input = tf.expand_dims(y_target[:, t], 1)

    batch_avg_loss = (loss / int(y_target.shape[1]))  # 计算 batch 的平均损失

    variables = encoder.trainable_variables + decoder.trainable_variables  # 整合所有变量
    gradients = tape.gradient(loss, variables)  # 计算各变量梯度
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_avg_loss



