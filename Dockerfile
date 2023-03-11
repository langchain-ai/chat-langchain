FROM google/cloud-sdk:latest


COPY . /app

# Bad idea should be passed as volume, ok for testing
COPY .env /app/.env
RUN rm -rf /app/aide.blank.app

WORKDIR /app

# check that vectorstore.pkl exist
RUN ls -l /app/vectorstore.pkl

# upgrade pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 9000
CMD python3 -m uvicorn main:app --host 0.0.0.0 --port 9000
