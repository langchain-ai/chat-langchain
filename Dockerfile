FROM python

ARG OPENAI_API_KEY

ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run the ingest script to load the LangChain docs data into the vectorstore
RUN sh ingest.sh

# Expose port 9000 for the app
EXPOSE 9000

# Start the app
CMD ["python", "main.py"]
