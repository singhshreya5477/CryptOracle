# 🚀 Quick Start Guide - CryptoSense

## Complete Setup (First Time)

### 1. Environment Setup
```powershell
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```powershell
# This will collect data, engineer features, train model
python run_pipeline.py
```

**Expected time:** ~15-25 minutes

### 3. Launch Dashboard
```powershell
streamlit run dashboard.py
```

Open browser to `http://localhost:8501`

---

## Individual Steps (If you prefer step-by-step)

### Step 1: Collect Data
```powershell
python data_collection.py
```
- Fetches BTC & ETH historical prices
- Gets Fear & Greed Index
- Retrieves CoinGecko market data
- Saves to `data/raw/`

### Step 2: Engineer Features
```powershell
python feature_engineering.py
```
- Creates 60+ technical indicators
- RSI, MACD, Bollinger Bands, etc.
- Saves to `data/processed/`

### Step 3: Preprocess Data
```powershell
python preprocessing.py
```
- Scales features
- Creates 60-day sequences
- Splits into train/val/test
- Saves `.npz` files

### Step 4: Train Model
```powershell
python train.py
```
- Builds Bidirectional LSTM + Attention
- ~100 epochs with early stopping
- Saves best model to `models/saved_models/`
- Creates training plots

### Step 5: View Results
```powershell
streamlit run dashboard.py
```

---

## Explore the Data (Optional)

Open `exploration.ipynb` in Jupyter:
```powershell
jupyter notebook exploration.ipynb
```
or
```powershell
code exploration.ipynb  # if using VS Code
```

---

## Troubleshooting

### "No module named X"
```powershell
pip install -r requirements.txt
```

### "Data not found"
Run in order:
1. `python data_collection.py`
2. `python feature_engineering.py`
3. `python preprocessing.py`
4. `python train.py`

### API Errors
- Check internet connection
- Wait 1-2 minutes (rate limiting)
- APIs used are free, no keys needed

### Training too slow?
Edit `config.py`:
- Reduce `EPOCHS` from 100 to 50
- Reduce `LOOKBACK_WINDOW` from 60 to 30
- Use GPU if available (TensorFlow auto-detects)

---

## Project Structure

```
CryptOracle/
├── config.py                 # Settings & hyperparameters
├── data_collection.py        # Step 1: Fetch data
├── feature_engineering.py    # Step 2: Create features
├── preprocessing.py          # Step 3: Prepare data
├── model.py                  # LSTM architecture
├── train.py                  # Step 4: Train model
├── evaluation.py             # Metrics & evaluation
├── dashboard.py              # Step 5: Interactive UI
├── advanced_features.py      # Anomaly, regime, Monte Carlo
├── run_pipeline.py           # Run all steps
├── exploration.ipynb         # Data exploration
├── requirements.txt          # Dependencies
└── README.md                 # Full documentation
```

---

## What You Get

After running the pipeline, you'll have:

✅ Raw data (CSV files)
✅ Featured data with 60+ indicators
✅ Trained LSTM model with attention
✅ Training history plots
✅ Evaluation metrics & reports
✅ Anomaly & regime detection plots
✅ Monte Carlo simulation
✅ Interactive dashboard

---

## Next Steps

1. **Experiment**: Modify hyperparameters in `config.py`
2. **Extend**: Add more cryptocurrencies
3. **Improve**: Try different architectures
4. **Deploy**: Host dashboard on Streamlit Cloud
5. **Share**: Show it in interviews!

---

## Tips for Interviews

When presenting this project, highlight:

1. **Multi-source data integration** (price + sentiment + on-chain)
2. **Attention mechanism** (publishable-level innovation)
3. **Practical metrics** (directional accuracy for trading)
4. **Advanced features** (anomaly, regime, Monte Carlo)
5. **Production-ready** (modular code, documentation, dashboard)

**What I learned:**
- Time series data requires special handling (no shuffling!)
- Feature engineering matters more than model complexity
- Always compare against a baseline
- Visualizations help communicate results

---

## Support

Found a bug? Have questions?
- Check `README.md` for detailed docs
- Review code comments
- Open an issue on GitHub

---

Happy forecasting! 🚀📈
