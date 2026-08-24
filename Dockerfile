FROM python:3.11-slim

WORKDIR /app

# install dependences
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the code
COPY . .

EXPOSE 8000

# start the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]