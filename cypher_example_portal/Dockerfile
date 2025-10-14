FROM python:3.11-slim
WORKDIR /app

COPY ./src/ /app/

COPY ./pyproject.toml /code/pyproject.toml
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir /code/.

RUN sed -i '1s/^\xEF\xBB\xBF//' /app/entrypoint.sh \
 && sed -i 's/\r$//' /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8502

EXPOSE 8502
CMD ["sh", "/app/entrypoint.sh"]
