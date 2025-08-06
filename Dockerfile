FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ADD . /app
RUN uv sync --frozen
# 暴露端口
EXPOSE 8189
# # 启动命令（根据实际应用调整）
CMD ["uv", "--directory", "src/server/", "run", "main.py", "-p", "8189", "-t", "sse", "-l", "debug"]
