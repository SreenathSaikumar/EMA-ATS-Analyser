FROM python:3.12.3-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install poetry==1.8.2

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction

COPY . .

RUN rm -f .env

CMD ["sh", "start.sh"]