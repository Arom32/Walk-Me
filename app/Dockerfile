#  Python 3.12 리눅스 베이스 이미지 설정
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 필수 리눅스 패키지 설치
# psycopg2 빌드에 필요한 libpq-dev, gcc 등 포함
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치 (캐싱 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 실행 (FastAPI 기준)
# --proxy-headers: 리눅스 서버(Nginx 등) 뒤에서 돌릴 때 필요
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]