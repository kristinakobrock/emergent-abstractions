FROM python:3.12

# Set up the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock ./

# Install the project dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Create a non-root user
RUN useradd -ms /bin/bash devuser
USER devuser

# Set up the entrypoint
CMD ["bash"]