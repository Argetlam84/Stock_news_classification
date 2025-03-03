
<p align="center">
  <img src="https://github.com/user-attachments/assets/8bc88860-2a1d-4ece-97e4-a30a494556a8" alt="Image description" />
</p>

# Stock News Analysis Project

This project leverages natural language processing (NLP) and machine learning models to classify stock market news. It began with thousands of labeled sentences related to the stock market obtained from Hugging Face and evolved step-by-step by asking, "How can I take this further?" The result is an automated project scheduled to run on GitHub Actions.

## <a href="https://stocknewsclassification.streamlit.app/">Project</a> Structure

The project is organized into several key folders, including **notebooks** and **models**:
- **notebooks**: Contains experimental work and visualizations of analyses performed throughout the project. This folder may have a complex structure as it includes different model trials and detailed data inspections.
- **models**: Stores models generated on the 1st of each month. Currently, three models are included: Logistic Regression, XGBRFClassifier, and LSTM.

## Workflow and Scheduling

The project is set to run every 4 hours using GitHub Actions:
1. **`web_scrapper.py`**: Scrapes data from [Yahoo Finance](https://finance.yahoo.com/topic/stock-market-news/).
2. **`news_clustering.py`**: Labels the scraped news using K-Means clustering. The labels are:
   - 0: Negative
   - 1: Neutral
   - 2: Positive
3. **`combine.py`**: Merges the labeled news with the existing `sentences.csv` file.

On the 1st of each month, **Logistic Regression**, **XGBRFClassifier**, and **LSTM** models are trained and saved to the `models` folder.

## Model Selection and Usage

You can experiment with these models through a Streamlit interface. The **LSTM model** currently exhibits **overfitting** due to its complex structure and limited data availability. This issue is expected to improve as more data is collected.

## Requirements

The following Python libraries are required for the project:
- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `spaCy`
- `xgboost`
- `optuna`
- `imbalanced-learn`
- `beautifulsoup4`
- `selenium`
- `webdriver-manager`
- `tensorflow`
- `streamlit`

To install requirements, run:
```bash
pip install -r requirements.txt

