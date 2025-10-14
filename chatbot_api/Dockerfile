FROM python:3.11-slim
WORKDIR /app

# Code under /app/src so imports like "src.*" work
COPY ./src/ /app/src
ENV PYTHONPATH=/app/src

# Install package
COPY ./pyproject.toml /code/pyproject.toml
RUN echo "==== /code/pyproject.toml ====" && sed -n '1,200p' /code/pyproject.toml
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir "langchain==0.2.15" "langchain-community==0.2.15" "langchain-openai>=0.1.20" "openai>=1.36.0" \
 && pip install --no-cache-dir /code/.


# Sanitize and chmod the script that is inside src/
RUN sed -i '1s/^\xEF\xBB\xBF//' /app/src/entrypoint.sh \
 && sed -i 's/\r$//' /app/src/entrypoint.sh \
 && chmod +x /app/src/entrypoint.sh

EXPOSE 8000
CMD ["sh", "/app/src/entrypoint.sh"]
