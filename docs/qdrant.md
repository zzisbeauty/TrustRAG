
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


## 参考资料

- 官方教程：https://qdrant.tech/documentation/beginner-tutorials/search-beginners/

