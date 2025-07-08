# Portfolio Risk Forecasting Dashboard

This Streamlit app forecasts portfolio risk using:
- **GARCH** for volatility forecasting
- **Historical & Monte Carlo VaR** for Value at Risk

## Files

- `app.py`: Streamlit application  
- `requirements.txt`: Python dependencies  
- `README.md`: Project overview  
- `.gitignore`: files to ignore in Git

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
````

## Features

1. Fetches asset prices (via Yahoo Finance)
2. Calculates daily portfolio returns
3. Forecasts next-day volatility (GARCH)
4. Computes Historical & Monte Carlo VaR
5. Visualizes returns with VaR thresholds

