import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import datetime as dt

def fetch_data():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("https://finance.yahoo.com/topic/stock-market-news/")

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
    visited_links = set()  # To keep track of already visited links

    for _ in range(5):  
        scroll_down()

    div_list = driver.find_elements(By.XPATH, "//*[@id='Fin-Stream']//*[@class='Cf']")
    print(f"Total {len(div_list)} news found.")

    index = 0

    while index < len(div_list):
        try:
            div = div_list[index]
            link = div.find_element(By.TAG_NAME, 'a').get_attribute("href")

            # Skip premium and already visited articles
            if "premium" not in link and link not in visited_links:
                visited_links.add(link)  # Track visited links
                driver.get(link)  # Navigate to the article
                time.sleep(2)  

                article_div = wait_for_element("//*[contains(@class, 'body')]")
                if article_div:
                    paragraphs = article_div.find_elements(By.TAG_NAME, 'p')
                    content = "\n".join([p.text for p in paragraphs])
                    print(content[:200])  # Print the first 200 characters for debugging
                    data.append({"News": content})

                else:
                    print("Article content not found or did not load.")

                driver.back()
                time.sleep(2)  

                # Update div_list to get any new articles loaded
                scroll_down()
                div_list = driver.find_elements(By.XPATH, "//*[@id='Fin-Stream']//*[@class='Cf']")

                index += 1
            else:
                index += 1  

        except Exception as e:
            print("Error:", e)
            index += 1  

    driver.quit()

    df = pd.DataFrame(data)
    df.to_csv(f"datasets/stock_market_news_{today}.csv", index=False)
    print("Data saved to CSV file.")

    return df