import os
import re
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Configure Selenium (headless mode for faster scraping)
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

def clean_title(title):
    """Remove non-word characters except spaces, allowing Chinese and alphabetic characters."""
    return re.sub(r'[^\w\s]', '', title, flags=re.UNICODE).strip()

def save_pdf_locally(pdf_content, filename, folder):
    """Save PDF content to a local file in the specified folder."""
    try:
        # Ensure the folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = os.path.join(folder, filename)
        with open(file_path, 'wb') as f:
            f.write(pdf_content)
        print(f"Saved {filename} to {folder}.")
    except Exception as e:
        print(f"Failed to save {filename} to {folder}. Error: {e}")

def download_pdf(url, filename, folder):
    """Download the PDF from the given URL and save it locally."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        save_pdf_locally(response.content, filename, folder)
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")

def get_pdf_from_publication(publication_url):
    """Extract the PDF link from the publication page."""
    try:
        driver.get(publication_url)
        time.sleep(5)
        publication_soup = BeautifulSoup(driver.page_source, "lxml")
        base_url = "https://rpc.cfainstitute.org"
        pdf_url = None
        links = publication_soup.find_all("a", href=True)
        for link in links:
            href = link["href"]
            if href.lower().endswith(".pdf"):
                if not href.startswith("http"):
                    href = f"{base_url}{href}"
                pdf_url = href
                break
        return pdf_url
    except Exception as e:
        print(f"Error accessing {publication_url}: {e}")
        return None

def scrape_page_for_pdfs(page_url, downloaded_pdfs, max_pdfs, folder):
    """Scrape all publications on a specific page to download PDFs."""
    driver.get(page_url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "lxml")
    publications = soup.select(".coveo-result-cell")

    if not publications:
        print(f"No more publications found at {page_url}")
        return False

    for pub in publications:
        if len(downloaded_pdfs) >= max_pdfs:
            return False  # Reached the desired number of PDFs

        title_elem = pub.select_one(".CoveoResultLink")
        body_elem = pub.select_one(".result-body")

        if title_elem:
            title = title_elem.text.strip()
            pub_url = title_elem["href"]

            # Print title and publication URL once
            print(f"Title: {title}")
            print(f"Publication URL: {pub_url}")

            cleaned_title = clean_title(title)
            pdf_url = get_pdf_from_publication(pub_url)

            if pdf_url and pdf_url not in downloaded_pdfs:
                pdf_filename = f"{cleaned_title}.pdf"
                download_pdf(pdf_url, pdf_filename, folder)

                downloaded_pdfs.add(pdf_url)

    return True

def scrape_all_pages_for_pdfs():
    """Scrape pages until 3 PDFs are downloaded."""
    base_url = "https://rpc.cfainstitute.org/en/research-foundation/publications"
    page_number = 0
    downloaded_pdfs = set()
    max_pdfs = 3  # Download only 3 PDFs
    download_folder = "downloaded_pdfs"  # Folder to save PDFs

    while len(downloaded_pdfs) < max_pdfs:
        page_url = f"{base_url}#first={page_number * 10}&sort=%40officialz32xdate%20descending"
        print(f"Scraping: {page_url}")
        if not scrape_page_for_pdfs(page_url, downloaded_pdfs, max_pdfs, download_folder):
            break
        page_number += 1

# Wrap the main function in an entry point
if __name__ == "__main__":
    # For local run
    driver = webdriver.Chrome(options=chrome_options)
    # For Docker container run, comment out the above line and uncomment the line below
    # driver = webdriver.Remote(command_executor='http://selenium-chrome:4444/wd/hub', options=chrome_options)
    try:
        scrape_all_pages_for_pdfs()
    finally:
        driver.quit()
