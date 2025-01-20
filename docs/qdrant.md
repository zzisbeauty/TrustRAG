>TrustRAG项目地址🌟：**[https://github.com/gomate-community/TrustRAG](https://github.com/gomate-community/TrustRAG)**

>可配置的模块化RAG框架

## 环境依赖
本教程基于docker安装Qdrant数据库，在此之前请先安装docker.

- Docker - The easiest way to use Qdrant is to run a pre-built Docker image.
- Python version >=3.8

## 启动Qdrant容器
1.拉取镜像
```bash
docker pull qdrant/qdrant
```
2.启动qdrant容器服务

```bash
docker run -d \
    --name qdrant_server \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    -p 6333:6333 \
    qdrant/qdrant
```

- 从 `qdrant/qdrant` 镜像创建一个名为 `qdrant_server` 的容器。
- 将宿主机的 `$(pwd)/qdrant_storage` 目录挂载到容器的 `/qdrant/storage` 目录，以实现数据持久化。
- 将宿主机的 `6333` 端口映射到容器的 `6333` 端口，以便通过宿主机访问 Qdrant 服务。
- 容器在后台运行，不会占用当前终端。

```bash
docker logs qdrant_server
```
可以看到下面日志：
![](https://i-blog.csdnimg.cn/direct/9d0cc450ce534d8d984788ce08c3bc1e.png)
通过 http://localhost:6333/dashboard 地址访问web ui
![](https://i-blog.csdnimg.cn/direct/9a0d77c43f0249fb9d41d2b34950eadc.png)
## 基于RESTful API 操作向量数据库
### 第一步：创建一个集合
>Qdrant向量数据库的集合概念可以类比MYSQL的表结构，用于统一存储同一类向量数据，集合中存储的每一条数据，在Qdrant中称为点（points），这里的点有数学几何空间的点类似的意思，代表向量在几何空间中的表示（你就当成一条数据看待就行）。


首先，我们需要创建一个名为 `star_charts` 的集合，用来存储殖民地数据。每个位置都会用一个四维向量来表示，并且我们会使用点积（Dot Product）作为相似度搜索的距离度量。

运行以下命令来创建集合：

```json
PUT collections/star_charts
{
  "vectors": {
    "size": 4,
    "distance": "Dot"
  }
}
```

### 第二步：将数据加载到集合中
>创建好集合之后，我们可以向集合添加向量数据，在Qdrant中向量数据使用point表示，一条point数据包括三部分id、payload(关联数据)、向量数据（vector）三部分。


现在集合已经设置好了，接下来我们添加一些数据。每个位置都会有一个向量和一些额外的信息（称为 payload），比如它的名字。

运行以下请求来添加数据：

```json
PUT collections/star_charts/points
{
  "points": [
    {
      "id": 1,
      "vector": [0.05, 0.61, 0.76, 0.74],
      "payload": {
        "colony": "Mars"
      }
    },
    {
      "id": 2,
      "vector": [0.19, 0.81, 0.75, 0.11],
      "payload": {
        "colony": "Jupiter"
      }
    },
    {
      "id": 3,
      "vector": [0.36, 0.55, 0.47, 0.94],
      "payload": {
        "colony": "Venus"
      }
    },
    {
      "id": 4,
      "vector": [0.18, 0.01, 0.85, 0.80],
      "payload": {
        "colony": "Moon"
      }
    },
    {
      "id": 5,
      "vector": [0.24, 0.18, 0.22, 0.44],
      "payload": {
        "colony": "Pluto"
      }
    }
  ]
}
```

### 第三步：运行搜索查询
现在，我们来搜索一下与某个特定向量（代表一个空间位置）最接近的三个殖民地。这个查询会返回这些殖民地以及它们的 payload 信息。

运行以下查询来找到最近的殖民地：

```json
POST collections/star_charts/points/search
{
  "vector": [0.2, 0.1, 0.9, 0.7],
  "limit": 3,
  "with_payload": true
}
```

这样，你就可以找到与给定向量最接近的三个殖民地了！
![](https://i-blog.csdnimg.cn/direct/876f3dc307f149e995c5f0bb52b18760.png)

---

上面命令，我们都可以在面板里面执行，
![](https://i-blog.csdnimg.cn/direct/840bb62d209140cebfcfd781122fadf8.png)
点击集合可以看到我们刚刚创建的例子：
![](https://i-blog.csdnimg.cn/direct/fe5eb121f3aa492791c3ca35453325dc.png)
点击可视化，我们可以看到集合里面的向量(point)
![](https://i-blog.csdnimg.cn/direct/cd3ccf1711914af3a2d79729d8eb4b14.png)
更多高级用法可以查看面板中的教程：
>http://localhost:6333/dashboard#/tutorial

## 基于qdrant_client操作向量数据库
以下是将上述内容转换为 Markdown 格式的版本：

```markdown
# Qdrant 快速入门指南

## 安装 `qdrant-client` 包（Python）

```bash
pip install qdrant-client
```

### 初始化客户端

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
```

### 创建 Collection

所有的向量数据（vector data）都存储在 Qdrant Collection 上。创建一个名为 `test_collection` 的 collection，该 collection 使用 `dot product` 作为比较向量的指标。

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=4, distance=Distance.DOT),
)
```

### 添加带 Payload 的向量

Payload 是与向量相关联的数据。

```python
from qdrant_client.models import PointStruct

operation_info = client.upsert(
    collection_name="test_collection",
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ]
)

print(operation_info)
```

### 运行查询

```python
search_result = client.query_points(
    collection_name="test_collection", query=[0.2, 0.1, 0.9, 0.7], limit=3
).points

print(search_result)
```

### 输出

```json
[
  {
    "id": 4,
    "version": 0,
    "score": 1.362,
    "payload": null,
    "vector": null
  },
  {
    "id": 1,
    "version": 0,
    "score": 1.273,
    "payload": null,
    "vector": null
  },
  {
    "id": 3,
    "version": 0,
    "score": 1.208,
    "payload": null,
    "vector": null
  }
]
```

### 添加过滤器

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

search_result = client.query_points(
    collection_name="test_collection",
    query=[0.2, 0.1, 0.9, 0.7],
    query_filter=Filter(
        must=[FieldCondition(key="city", match=MatchValue(value="London"))]
    ),
    with_payload=True,
    limit=3,
).points

print(search_result)
```


```json
[
    {
        "id": 2,
        "version": 0,
        "score": 0.871,
        "payload": {
            "city": "London"
        },
        "vector": null
    }
]
```


## 语义搜索入门实现
以官方教程为例，我在TrustRAG中对Qdrant进行了封装改造：

>官方教程：[https://qdrant.tech/documentation/beginner-tutorials/neural-search/](https://qdrant.tech/documentation/beginner-tutorials/neural-search/)
>TrusRAG实现代码`QdrantEngine`:[https://github.com/gomate-community/TrustRAG/blob/main/trustrag/modules/engine/qdrant.py](https://github.com/gomate-community/TrustRAG/blob/main/trustrag/modules/engine/qdrant.py)

以下为使用完整代码：
```python
from trustrag.modules.engine.qdrant import QdrantEngine
from trustrag.modules.engine.qdrant import SentenceTransformerEmbedding
if __name__ == "__main__":
    # Initialize embedding generators
    local_embedding_generator = SentenceTransformerEmbedding(model_name_or_path="all-MiniLM-L6-v2", device="cpu")
    # openai_embedding_generator = OpenAIEmbedding(api_key="your_key", base_url="https://ark.cn-beijing.volces.com/api/v3", model="your_model_id")

    # Initialize QdrantEngine with local embedding generator
    qdrant_engine = QdrantEngine(
        collection_name="startups",
        embedding_generator=local_embedding_generator,
        qdrant_client_params={"host": "192.168.1.5", "port": 6333},
    )

    documents=[
        {"name": "SaferCodes", "images": "https:\/\/safer.codes\/img\/brand\/logo-icon.png",
         "alt": "SaferCodes Logo QR codes generator system forms for COVID-19",
         "description": "QR codes systems for COVID-19.\nSimple tools for bars, restaurants, offices, and other small proximity businesses.",
         "link": "https:\/\/safer.codes", "city": "Chicago"},
        {"name": "Human Practice",
         "images": "https:\/\/d1qb2nb5cznatu.cloudfront.net\/startups\/i\/373036-94d1e190f12f2c919c3566ecaecbda68-thumb_jpg.jpg?buster=1396498835",
         "alt": "Human Practice -  health care information technology",
         "description": "Point-of-care word of mouth\nPreferral is a mobile platform that channels physicians\u2019 interest in networking with their peers to build referrals within a hospital system.\nHospitals are in a race to employ physicians, even though they lose billions each year ($40B in 2014) on employment. Why ...",
         "link": "http:\/\/humanpractice.com", "city": "Chicago"},
        {"name": "StyleSeek",
         "images": "https:\/\/d1qb2nb5cznatu.cloudfront.net\/startups\/i\/3747-bb0338d641617b54f5234a1d3bfc6fd0-thumb_jpg.jpg?buster=1329158692",
         "alt": "StyleSeek -  e-commerce fashion mass customization online shopping",
         "description": "Personalized e-commerce for lifestyle products\nStyleSeek is a personalized e-commerce site for lifestyle products.\nIt works across the style spectrum by enabling users (both men and women) to create and refine their unique StyleDNA.\nStyleSeek also promotes new products via its email newsletter, 100% personalized ...",
         "link": "http:\/\/styleseek.com", "city": "Chicago"},
        {"name": "Scout",
         "images": "https:\/\/d1qb2nb5cznatu.cloudfront.net\/startups\/i\/190790-dbe27fe8cda0614d644431f853b64e8f-thumb_jpg.jpg?buster=1389652078",
         "alt": "Scout -  security consumer electronics internet of things",
         "description": "Hassle-free Home Security\nScout is a self-installed, wireless home security system. We've created a more open, affordable and modern system than what is available on the market today. With month-to-month contracts and portable devices, Scout is a renter-friendly solution for the other ...",
         "link": "http:\/\/www.scoutalarm.com", "city": "Chicago"},
        {"name": "Invitation codes", "images": "https:\/\/invitation.codes\/img\/inv-brand-fb3.png",
         "alt": "Invitation App - Share referral codes community ",
         "description": "The referral community\nInvitation App is a social network where people post their referral codes and collect rewards on autopilot.",
         "link": "https:\/\/invitation.codes", "city": "Chicago"},
        {"name": "Hyde Park Angels",
         "images": "https:\/\/d1qb2nb5cznatu.cloudfront.net\/startups\/i\/61114-35cd9d9689b70b4dc1d0b3c5f11c26e7-thumb_jpg.jpg?buster=1427395222",
         "alt": "Hyde Park Angels - ",
         "description": "Hyde Park Angels is the largest and most active angel group in the Midwest. With a membership of over 100 successful entrepreneurs, executives, and venture capitalists, the organization prides itself on providing critical strategic expertise to entrepreneurs and ...",
         "link": "http:\/\/hydeparkangels.com", "city": "Chicago"},
        {"name": "GiveForward",
         "images": "https:\/\/d1qb2nb5cznatu.cloudfront.net\/startups\/i\/1374-e472ccec267bef9432a459784455c133-thumb_jpg.jpg?buster=1397666635",
         "alt": "GiveForward -  health care startups crowdfunding",
         "description": "Crowdfunding for medical and life events\nGiveForward lets anyone to create a free fundraising page for a friend or loved one's uncovered medical bills, memorial fund, adoptions or any other life events in five minutes or less. Millions of families have used GiveForward to raise more than $165M to let ...",
         "link": "http:\/\/giveforward.com", "city": "Chicago"},
        {"name": "MentorMob",
         "images": "https:\/\/d1qb2nb5cznatu.cloudfront.net\/startups\/i\/19374-3b63fcf38efde624dd79c5cbd96161db-thumb_jpg.jpg?buster=1315734490",
         "alt": "MentorMob -  digital media education ventures for good crowdsourcing",
         "description": "Google of Learning, indexed by experts\nProblem: Google doesn't index for learning. Nearly 1 billion Google searches are done for \"how to\" learn various topics every month, from photography to entrepreneurship, forcing learners to waste their time sifting through the millions of results.\nMentorMob is ...",
         "link": "http:\/\/www.mentormob.com", "city": "Chicago"},
        {"name": "The Boeing Company",
         "images": "https:\/\/d1qb2nb5cznatu.cloudfront.net\/startups\/i\/49394-df6be7a1eca80e8e73cc6699fee4f772-thumb_jpg.jpg?buster=1406172049",
         "alt": "The Boeing Company -  manufacturing transportation", "description": "",
         "link": "http:\/\/www.boeing.com", "city": "Berlin"},
        {"name": "NowBoarding \u2708\ufe0f",
         "images": "https:\/\/static.above.flights\/img\/lowcost\/envelope_blue.png",
         "alt": "Lowcost Email cheap flights alerts",
         "description": "Invite-only mailing list.\n\nWe search the best weekend and long-haul flight deals\nso you can book before everyone else.",
         "link": "https:\/\/nowboarding.club\/", "city": "Berlin"},
        {"name": "Rocketmiles",
         "images": "https:\/\/d1qb2nb5cznatu.cloudfront.net\/startups\/i\/158571-e53ddffe9fb3ed5e57080db7134117d0-thumb_jpg.jpg?buster=1361371304",
         "alt": "Rocketmiles -  e-commerce online travel loyalty programs hotels",
         "description": "Fueling more vacations\nWe enable our customers to travel more, travel better and travel further. 20M+ consumers stock away miles & points to satisfy their wanderlust.\nFlying around or using credit cards are the only good ways to fill the stockpile today. We've built the third way. Customers ...",
         "link": "http:\/\/www.Rocketmiles.com", "city": "Berlin"}

    ]
    vectors = qdrant_engine.embedding_generator.generate_embedding([doc["description"] for doc in documents])
    print(vectors.shape)
    payload = [doc for doc  in documents]

    # Upload vectors and payload
    qdrant_engine.upload_vectors(vectors=vectors, payload=payload)

    # Build a filter for city and category
    conditions = [
        {"key": "city", "match": "Berlin"},
    ]
    custom_filter = qdrant_engine.build_filter(conditions)

    # Search for startups related to "vacations" in Berlin
    results = qdrant_engine.search(text="vacations", query_filter=custom_filter, limit=5)
    for result in results:
        print(result)
```

## 参考资料

- 官方教程：[https://qdrant.tech/documentation/beginner-tutorials/search-beginners/](https://qdrant.tech/documentation/beginner-tutorials/search-beginners/)
- Qdrant向量数据库介绍：[https://www.tizi365.com/topic/8144.html](https://www.tizi365.com/topic/8144.html)
- Qdrant官方快速入门和教程简化版：[https://www.cnblogs.com/shizidushu/p/18385637](https://www.cnblogs.com/shizidushu/p/18385637)
- 【RAG利器】向量数据库qdrant各种用法，多种embedding生成方法 
：[https://www.cnblogs.com/zxporz/p/18336698](https://www.cnblogs.com/zxporz/p/18336698)

