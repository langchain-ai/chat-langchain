import os
from io import BytesIO

import PyPDF2
import boto3
from langchain.chat_models import ChatOpenAI


def initialize_llm(model):
    return ChatOpenAI(
        model=model,
        streaming=True,
        temperature=0,
    )


def read_pdf_from_s3(bucket_name, file_key):
    # Initialize S3 client
    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

    # Download PDF file from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    pdf_data = response['Body'].read()

    # Read text content from the PDF
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_data))
    text_content = ''
    for page_num in range(len(pdf_reader.pages)):
        text_content += pdf_reader.pages[page_num].extract_text()

    return text_content


def remove_long_words(text, character_len=30):
    # Split the text into words
    words = text.split()

    # Remove words longer than 30 characters
    filtered_words = [word for word in words if len(word) <= character_len]

    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)

    return filtered_text
