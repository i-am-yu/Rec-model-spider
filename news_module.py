import time
import keras 
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding, Attention, Concatenate
import jieba
import tensorflow as tf
import numpy as np
from common_module import *
from sina.models.newsEmbedding import NewsEmbedding
from sklearn.metrics.pairwise import cosine_similarity


# 注意：删除了原有的 Encoder 和 Decoder 类，因为我们现在使用扁平化结构

class NewsUtils(object):
    def __init__(self, load_model=True):
        print("加载用户字典数据。。。")
        jieba.load_userdict(get_val(DICT_PATH))
        print("用户字典数据加载完成。。。")
        self.word2idx = get_val(WORD2IDX)
        self.idx2word = get_val(IDX2WORD)
        self.word_embedding_martix = get_val(WORD_EMBEDDING_MATRIX)
        self.news_model = None
        self.encoder_model = None  # 新增：专门用于生成向量的子模型
        self.max_len = config["modelConfig"]["titleMaxLen"]
        if load_model:
            self.load_news_model()

    def cut_titles(self, news_titles):
        print("标题分词处理...")
        cut_result = []
        for title in news_titles:
            cut_result.append(jieba.lcut(title.strip()))
        print("标题分词完成...")
        return cut_result

    def get_title_seq(self, cut_news_titles):
        title_seq = []
        for news_title in cut_news_titles:
            words_idx = [self.word2idx.get(word, 3) for word in news_title]  # 3 is <UNK>
            title_seq.append(words_idx)

        title_seq = keras.preprocessing.sequence.pad_sequences(sequences=title_seq,
                                                               maxlen=self.max_len,
                                                               padding="post")
        return title_seq

    def cal_similar_news(self, doc_id):
        # 计算相似新闻
        try:
            target_news = list(NewsEmbedding.objects(doc_id=doc_id))[0]
        except IndexError:
            return []

        target_embedding = np.array(list(target_news["embedding"])).reshape((1, 200))
        # 读取配置中的天数，默认7天
        days = config["modelConfig"].get("similarNewsDay", 7)
        last_time = int(time.time()) - (days * 86400)

        news_embedding_obj = list(NewsEmbedding.objects(create_time__gt=last_time))

        if not news_embedding_obj:
            return []

        news_embedding_list = []
        doc_ids = []
        for news_embedding in news_embedding_obj:
            news_embedding_list.append(list(news_embedding["embedding"]))
            doc_ids.append(news_embedding["doc_id"])

        # 批量计算余弦相似度
        news_similar_score = cosine_similarity(news_embedding_list, target_embedding).flatten()
        score_index = np.argsort(-news_similar_score)  # 降序排列

        news_sorted_list = []
        for news_index in score_index[:30]:  # 取前 30 篇
            similar_doc_id = doc_ids[news_index]
            if doc_id != similar_doc_id:
                news_sorted_list.append(similar_doc_id)
        return news_sorted_list

    def generate_title_embedding(self, news_titles):
        if self.encoder_model is None:
            print("错误：模型未加载，无法生成向量")
            return []

        print("准备生成标题向量...")
        cut_titles_seq = self.cut_titles(news_titles)
        title_seq = self.get_title_seq(cut_titles_seq)

        print(f"开始计算 {len(news_titles)} 条新闻的特征向量...")
        # 使用 predict 批量计算，比原来的 for 循环快得多
        # encoder_model 的输出就是 state_h (200维向量)
        news_embedding = self.encoder_model.predict(title_seq, batch_size=64)

        print("向量计算完成！")
        return news_embedding.tolist()

    def load_news_model(self):
        print("开始重建新闻模型结构（与训练代码一致）...")
        vocab_size = len(self.word_embedding_martix)
        latent_dim = 200

        # --- 1. Encoder 部分 ---
        enc_input = Input(shape=(self.max_len,), name="input_1")
        # 注意：训练时创建了两个 Embedding 实例，这里必须保持一致
        enc_emb = Embedding(vocab_size, 200, weights=[self.word_embedding_martix], trainable=False)(enc_input)

        enc_gru_layer = GRU(latent_dim, return_sequences=True, return_state=True)
        enc_outputs, state_h = enc_gru_layer(enc_emb)

        # --- 2. Decoder 部分 ---
        dec_input = Input(shape=(self.max_len,), name="input_2")
        dec_emb = Embedding(vocab_size, 200, weights=[self.word_embedding_martix], trainable=False)(dec_input)

        dec_gru_layer = GRU(latent_dim, return_sequences=True)
        dec_outputs = dec_gru_layer(dec_emb, initial_state=state_h)

        # --- 3. Attention & Output ---
        attn_layer = Attention()
        context_vector = attn_layer([dec_outputs, enc_outputs])
        decoder_combined_context = Concatenate(axis=-1)([context_vector, dec_outputs])

        outputs = Dense(vocab_size, activation='softmax')(decoder_combined_context)

        # 构建完整模型（用于加载权重）
        model = Model([enc_input, dec_input], outputs)

        print("模型结构构建完毕，正在加载 weights.h5 ...")
        try:
            model.load_weights(get_val(MODEL_WEIGHTS_PATH))
            print("权重加载成功！")
        except ValueError as e:
            print("权重加载失败！请确保 train_model.py 和这里的结构完全一致。")
            raise e

        self.news_model = model

        # --- 4. 构建特征提取子模型 ---
        # 我们只需要 Encoder 的输出 state_h (200维向量)
        # 这个模型共享了上面 model 的层和权重，所以不需要额外加载权重
        self.encoder_model = Model(inputs=enc_input, outputs=state_h)
        print("特征提取子模型准备就绪。")