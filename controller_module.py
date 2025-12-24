import time
from sina.models.news import News
from sina.models.newsEmbedding import NewsEmbedding
from flask import Flask, request
from response_entry import *
from spider_service import *
from multiprocessing import Process
from news_service import *

app = Flask(__name__)


@app.route("/runSpider")
def run_spider():
    spider_name = request.args.get("spiderName")
    crawl_num = request.args.get("crawlNum")
    if crawl_num is not None:
        crawl_num = int(crawl_num)
    p = Process(target=start_spider, args=(spider_name, crawl_num))
    p.start()
    return success()


@app.route("/spiderStatus")
def get_spider_status():
    spider_name = request.args.get("spiderName")
    spider_info = json.loads(rd.get(SPIDER_INFO(spider_name)))
    return success(spider_info)


@app.route('/setNewsEmbedding', methods=['GET', 'POST'])
def set_news_embedding():
    print("收到向量化请求，开始处理...")

    # 1. 获取工具类实例
    news_utils = get_val(NEWS_UTILS)
    if not news_utils:
        return "错误：NewsUtils 未初始化，请检查 main.py 是否开启了 load_model=True"

    # 2. 从数据库读取所有新闻 (替代原来的 request.get_data)
    print("正在从 MongoDB 读取所有新闻数据...")
    # 这里读取全部新闻，如果数据量特别大(>10万)可能需要分页，2万条直接读没问题
    news_list = News.objects.all()

    titles = []
    doc_ids = []
    for news in news_list:
        # 过滤掉空标题
        if news.title and len(news.title.strip()) > 0:
            titles.append(news.title)
            doc_ids.append(news.doc_id)

    if len(titles) == 0:
        return "数据库中没有新闻，无法计算！"

    print(f"共读取到 {len(titles)} 条有效新闻，开始调用模型计算向量...")

    # 3. 生成向量 (调用 news_utils 中的模型)
    # 这可能会花一点时间，请耐心等待
    try:
        embedding_list = news_utils.generate_title_embedding(titles)
    except Exception as e:
        print(f"计算出错: {str(e)}")
        return f"模型计算出错: {str(e)}"

    # 4. 保存结果到 NewsEmbedding 表
    print("计算完成，正在保存结果到数据库...")
    saved_count = 0
    # 为了防止写入太慢，我们可以先检查是否已经存在
    existing_ids = set(NewsEmbedding.objects.scalar('doc_id'))

    for i in range(len(doc_ids)):
        doc_id = doc_ids[i]

        # 如果已经有了，就跳过
        if doc_id in existing_ids:
            continue

        vec = embedding_list[i]
        # 插入新记录
        NewsEmbedding(doc_id=doc_id, embedding=vec, create_time=int(time.time())).save()
        saved_count += 1

    print(f"处理完成！新入库 {saved_count} 条向量。")
    return f"Success! Total scanned: {len(titles)}, New saved: {saved_count}"


@app.route("/setUserRecommenderList", methods=["POST"])
def set_user_recommender_list():
    """生成一个用户的推荐列表"""
    user_info = json.loads(request.get_data(), encoding="utf-8")
    generate_user_recommender_list(user_info)
    return success()


@app.route("/getSimilarNewsList")
def get_similar_news_list():
    doc_id = request.args.get("docId")
    res = similar_news_list(doc_id)
    return success({"similarNews": res})
