
## 环境依赖
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

## 参考资料

- 官方教程：https://qdrant.tech/documentation/beginner-tutorials/search-beginners/