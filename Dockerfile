# 使用官方 Python 运行时作为父镜像
FROM python:3.11

# 设置工作目录为 /app
WORKDIR /app

# 将当前目录内容复制到位于 /app 中的容器中
COPY . /app

# 通过 pip 命令安装任何所需的包
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# 使端口 80 可供此容器外的环境使用
EXPOSE 80

# 在容器启动时运行 app.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]