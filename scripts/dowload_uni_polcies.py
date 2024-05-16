import csv
import os

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def get_policies_pdfs():
    print("Fetching policies...")
    url = "https://www.wgtn.ac.nz/about/governance/policy/policies"
    # Set up the WebDriver (e.g., Chrome)
    driver = webdriver.Chrome()  # Ensure that chromedriver is in your PATH

    try:
        # Load the page
        driver.get(url)

        # Wait for the specific element to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "title-link"))
        )

        # Get the page source after waiting for the elements
        page_source = driver.page_source

        # Parse the HTML content with Beautiful Soup
        soup = BeautifulSoup(page_source, "html.parser")

        # Find all <a> tags with the specified attribute and class
        links = soup.find_all("a", {"data-v-0c921691": True, "class": "title-link"})

        # Extract the href attribute from each link
        pdf_links = [link["href"] for link in links if "href" in link.attrs]

        return pdf_links
    finally:
        # Close the WebDriver
        driver.quit()


def download_policy(pdf_link):
    response = requests.get(pdf_link)

    if response.status_code == 200:
        filename = pdf_link.split("/")[-1]
        # if the filename does not end in .pdf don't save it
        if filename.endswith(".pdf"):
            save_path = os.getcwd() + f"/data/vic-policies/{filename}"
            with open(save_path, "wb") as f:
                f.write(response.content)
                print(f"PDF saved to {save_path}")
        else:
            print(f"Skipping {filename} - not a PDF file.")
    else:
        print(f"Failed to download {pdf_link}")


def create_csv_index(pdf_links):
    with open("scripts/policies.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Policy Name", "URL"])
        for pdf_link in pdf_links:
            csv_writer.writerow([pdf_link.split("/")[-1], pdf_link])


pdf_links = get_policies_pdfs()
for pdf_link in pdf_links:
    download_policy(pdf_link)
create_csv_index(pdf_links)
