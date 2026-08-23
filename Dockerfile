FROM python:3.11-slim

# set up working directory
WORKDIR /app

# install dependences
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the code
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]