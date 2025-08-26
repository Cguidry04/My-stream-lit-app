# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime, time
from sklearn.ensemble import GradientBoostingClassifier
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import schedule
import threading

load_dotenv()

# -------------------------
# Env Variables
# -------------------------
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

alpaca_api = None
if ALPACA_KEY and ALPACA_SECRET:
    alpaca_api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version='v2')

# -------------------------
# Utilities
# -------------------------
def send_telegram(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url,data={"chat_id":TELEGRAM_CHAT_ID,"text":msg})
        except:
            pass

# -------------------------
# Indicators
# -------------------------
def calculate_indicators(df):
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100 - (100/(1+rs))
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2*df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2*df['Close'].rolling(20).std()
    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    df.drop(['H-L','H-PC','L-PC','TR','EMA_12','EMA_26'], axis=1, inplace=True)
    return df

# Candlestick patterns
def detect_candlestick(df):
    patterns=[]
    for i in range(len(df)):
        o,h,l,c = df.loc[i,["Open","High","Low","Close"]]
        body = abs(c-o)
        rng = h-l
        upper = h-max(o,c)
        lower = min(o,c)-l
        label = None
        if body <= 0.1*rng: label="Doji"
        elif lower >= 2*body and upper <= body: label="Hammer"
        if i>0:
            prev_o,prev_c = df.loc[i-1,["Open","Close"]]
            if c>o and prev_c<prev_o and c>prev_o and o<prev_c: label="Bullish Engulfing"
            elif c<o and prev_c>prev_o and c<prev_o and o>prev_c: label="Bearish Engulfing"
        patterns.append(label if label else "None")
    df['Pattern'] = patterns
    return df

# Liquidity detection
def detect_liquidity(df, window=5):
    levels=[]
    for i in range(window,len(df)-window):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        if high == max(df["High"].iloc[i-window:i+window+1]): levels.append(("resistance",high))
        if low == min(df["Low"].iloc[i-window:i+window+1]): levels.append(("support",low))
    cleaned=[]
    for lvl in levels:
        if not any(abs(lvl[1]-c[1])<0.5 for c in cleaned): cleaned.append(lvl)
    return cleaned

# -------------------------
# ML Predictor
# -------------------------
def train_ml(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['Close','Open','High','Low','Volume','SMA_5','SMA_20','EMA_5','EMA_20','RSI','MACD','Signal','BB_Mid','BB_Upper','BB_Lower','ATR']
    df.dropna(inplace=True)
    X = df[features]
    y = df['Target']
    model = GradientBoostingClassifier()
    model.fit(X,y)
    df['ML_Prob'] = model.predict_proba(X)[:,1]
    return df, model

# -------------------------
# Trade execution
# -------------------------
def execute_trade(symbol, qty, side, real=False):
    if alpaca_api:
        try:
            alpaca_api.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day")
            send_telegram(f"{'Live' if real else 'Paper'} trade executed: {side} {qty} {symbol}")
        except Exception as e:
            send_telegram(f"Trade Error: {e}")

# -------------------------
# Intraday Scheduler
# -------------------------
def run_scheduler(symbol, execute_real):
    def job():
        # Fetch latest 5m bars
        df = yf.download(symbol, period="7d", interval="5m").reset_index()
        df = calculate_indicators(df)
        df = detect_candlestick(df)
        df, model = train_ml(df)
        liquidity = detect_liquidity(df)
        
        balance = 10000
        position = 0
        
        for i,row in df.iterrows():
            buy_cond = row['SMA_5']>row['SMA_20'] and row['Pattern'] in ['Hammer','Bullish Engulfing'] and any(abs(row['Close']-lvl[1])<1 and lvl[0]=='support' for lvl in liquidity) and row['ML_Prob']>=0.7
            sell_cond = row['SMA_5']<row['SMA_20'] and row['Pattern'] in ['Doji','Bearish Engulfing'] and position>0
            if buy_cond and position==0:
                qty = int((balance*0.02)/row['ATR'])
                position = qty
                balance -= qty*row['Close']
                if execute_real: execute_trade(symbol, qty, "buy", real=True)
            elif sell_cond and position>0:
                balance += position*row['Close']
                if execute_real: execute_trade(symbol, position, "sell", real=True)
                position=0
        if position>0: balance += position*df.iloc[-1]['Close']
        send_telegram(f"Intraday Scheduler Completed: Portfolio Value ${balance:.2f}")
    
    schedule.every(5).minutes.do(job)
    
    def run_thread():
        while True:
            schedule.run_pending()
            time.sleep(30)
    
    thread = threading.Thread(target=run_thread)
    thread.start()

# -------------------------
# Streamlit App
# -------------------------
st.title("AI Stock Trading Bot")
symbol = st.text_input("Stock Symbol","AAPL")
execute_real = st.checkbox("Execute Real Trades", value=False)
st.button("Start Intraday Scheduler", on_click=lambda: run_scheduler(symbol, execute_real))
st.write("Bot is running! Monitor via Telegram or Streamlit dashboard.")
