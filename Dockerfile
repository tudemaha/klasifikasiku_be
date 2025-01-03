FROM python:3.10.16-slim-bullseye

WORKDIR /app

COPY . .

RUN touch .env
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["fastapi", "run", "main.py"]