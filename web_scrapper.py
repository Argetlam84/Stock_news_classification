import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import datetime as dt


MAX_RETRIES = 3

def retry(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(5)


def fetch_data():

    options = webdriver.ChromeOptions()
    #options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = "https://finance.yahoo.com/topic/stock-market-news/"
    retry(driver.get, url)

    print(driver.title)

    today = dt.datetime.today().strftime("%Y-%m-%d")

    def scroll_down():
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    def wait_for_element(xpath, timeout=10):
        try:
            element = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
            return element
        except Exception as e:
            print(f"Element could not be found: {xpath}")
            return None

    data = []
    visited_links = set()

    for _ in range(5):
        scroll_down()

    div_list = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//*[contains(@class, 'content')]")))
    print(f"Total {len(div_list)} news found.")

    index = 0
    while index < len(div_list):
        try:
            div = div_list[index]
            link = div.find_element(By.TAG_NAME, "a").get_attribute("href")

            if link and "premium" not in link and link not in visited_links:
                print(f"Visiting link: {link}")
                visited_links.add(link)
                retry(driver.get, link)
                time.sleep(2)

                article_div = wait_for_element("//*[contains(@class, 'body')]")
                if article_div:
                    paragraphs = article_div.find_elements(By.TAG_NAME, "p")
                    content = "\n".join([p.text for p in paragraphs])
                    print(f"Content fetched: {content[:200]}...")
                    data.append({"News": content})
                else:
                    print("Article content not found or did not load.")

                driver.back()
                time.sleep(2)

                scroll_down()
                div_list = driver.find_elements(By.XPATH, "//*[contains(@class, 'content')]")
            index += 1

        except Exception as e:
            print("Error:", e)
            index += 1

    driver.quit()

    df = pd.DataFrame(data)
    turkish_chars = "[çğıöşüÇĞİÖŞÜ]"
    df = df[~df["News"].str.contains(turkish_chars, regex=True, na=False)]
    output_path = f"datasets/stock_market_news_{today}.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to CSV file: {output_path}")

    return df

