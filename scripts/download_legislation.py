import csv
import os

import requests
from bs4 import BeautifulSoup

file_path = os.getcwd() + "/scripts/legislation.csv"


def download_legislation(act_name, year, url):
    response = requests.get(url)

    # <a href="/act/public/2020/0031/latest/096be8ed81ce0ab4.pdf" id="ctl00_Cnt_documentNavigationHeader_linkPdfDownload" title="170 pages">Print/Download PDF [1.0MB]</a>

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all the td elements with the class "resultsTitle"
        result = soup.find(
            "a", {"id": "ctl00_Cnt_documentNavigationHeader_linkPdfDownload"}
        )
        # Print the list of hrefs
        pdf_url = f"https://www.legislation.govt.nz{result['href']}"
        print(pdf_url)

        save_path = os.getcwd() + f"/data/govt-docs/{act_name} {year}.pdf"
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
                print(f"PDF saved to {save_path}")

    else:
        print("Failed to fetch the page")


with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        print(", ".join(row))
        download_legislation(row[0], row[1], row[2])
