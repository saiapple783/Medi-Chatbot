FROM python:3.11-slim
WORKDIR /app

# Copy code (includes src/entrypoint.sh -> /app/entrypoint.sh)
COPY ./src/ /app/

# Install package
COPY ./pyproject.toml /code/pyproject.toml
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir /code/.

# Sanitize and chmod the existing script
RUN sed -i '1s/^\xEF\xBB\xBF//' /app/entrypoint.sh \
 && sed -i 's/\r$//' /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh

CMD ["sh", "/app/entrypoint.sh"]
