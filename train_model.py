import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from mongoengine import connect  # 1. 新增导入
from common_module import *
from main import load_word_embedding
from sina.models.news import News
import jieba

# 初始化配置
init_common()
title_max_len = config["modelConfig"]["titleMaxLen"]  # 默认20


def prepare_training_data():
    print("正在从数据库提取标题并分词...")
    # 加载词向量索引映射
    _, word2idx, _ = load_word_embedding()

    # 从数据库获取新闻标题
    # 注意：这里只会从 'source' 别名对应的数据库(news_db)中读取
    # 如果你爬取的数据在 news_recommender，请确保下面的连接配置正确
    news_list = News.objects.only('title')

    # 过滤空标题
    titles = [n.title for n in news_list if n.title]
    print(f"成功加载 {len(titles)} 条新闻标题")

    if len(titles) == 0:
        print("警告：数据库中没有读取到新闻！请检查MongoDB中是否有数据。")
        return np.array([]), word2idx

    X = []
    for title in titles:
        # 分词并转为 ID 序列
        words = jieba.lcut(title)
        seq = [word2idx.get(w, word2idx["<UNK>"]) for w in words[:title_max_len]]
        # 填充 PAD
        seq = seq + [word2idx["<PAD>"]] * (title_max_len - len(seq))
        X.append(seq)

    return np.array(X), word2idx


def build_train_model(word_embedding_matrix, vocab_size):
    # 作者论文描述：200维隐藏层，GRU
    latent_dim = 200

    # Encoder
    enc_input = Input(shape=(title_max_len,))
    enc_emb = Embedding(vocab_size, 200, weights=[word_embedding_matrix], trainable=False)(enc_input)
    enc_gru = GRU(latent_dim, return_sequences=True, return_state=True)
    enc_outputs, state_h = enc_gru(enc_emb)

    # Decoder (AutoEncoder模式：目标是还原输入序列)
    dec_input = Input(shape=(title_max_len,))
    dec_emb = Embedding(vocab_size, 200, weights=[word_embedding_matrix], trainable=False)(dec_input)
    dec_gru = GRU(latent_dim, return_sequences=True)
    dec_outputs = dec_gru(dec_emb, initial_state=state_h)

    # Attention 层
    attn_layer = Attention()
    context_vector = attn_layer([dec_outputs, enc_outputs])
    decoder_combined_context = Concatenate(axis=-1)([context_vector, dec_outputs])

    # 输出层
    outputs = Dense(vocab_size, activation='softmax')(decoder_combined_context)

    model = Model([enc_input, dec_input], outputs)
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy')
    return model


if __name__ == "__main__":
    # 1. 初始化全局配置
    init_common()

    # 2. 注入配置路径 (防止 KeyError)
    set_val(WORD_EMBEDDING_PATH, config["dataPath"]["WORD_EMBEDDING_PATH"])
    set_val(MODEL_WEIGHTS_PATH, config["dataPath"]["MODEL_WEIGHTS_PATH"])
    set_val(DICT_PATH, config["dataPath"]["DICT_PATH"])

    # 3. 【关键修复】连接数据库
    # 必须连接名为 'source' 的数据库，因为 News 模型默认使用它
    connect("news_db", alias="source")
    # 为了保险，也可以连接 target 库
    connect("news_recommender", alias="target")

    print("数据库连接成功，开始加载资源...")

    # 4. 加载资源与数据
    # 注意：prepare_training_data 内部会再次调用 load_word_embedding，
    # 为了避免重复加载大文件，我们可以优化一下逻辑，但在现在的逻辑下先跑通为主。
    X_train, word2idx = prepare_training_data()

    if len(X_train) == 0:
        print("数据为空，停止训练。")
        exit()

    # 获取词向量矩阵用于构建模型
    word_matrix, _, _ = load_word_embedding()
    vocab_size = len(word2idx)

    model = build_train_model(word_matrix, vocab_size)
    model.summary()

    print("开始训练模型...")
    # 因为是自编码器，目标 y 就是 X_train 本身 (需要增加维度匹配 sparse_categorical_crossentropy)
    model.fit([X_train, X_train], np.expand_dims(X_train, -1),
              batch_size=64,
              epochs=5,  # 可以先试跑 5 个 epoch
              validation_split=0.1)

    # 保存权重
    save_path = config["dataPath"]["MODEL_WEIGHTS_PATH"]
    model.save_weights(save_path)
    print(f"训练完成，模型已保存至: {save_path}")