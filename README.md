# TradePilot - Automated Forex & Crypto Trading Bot

An advanced algorithmic trading system that combines multiple trading strategies, machine learning validation, and a modern React dashboard for real-time monitoring and control.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         TradePilot                              │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React + Vite)           │  Backend (Flask + Python)  │
│  ─────────────────────────         │  ────────────────────────  │
│  • Dashboard                       │  • REST API (app.py)       │
│  • Trade History                   │  • Trading Bot (main.py)   │
│  • Analytics                       │  • Strategy Engine         │
│  • Settings                        │  • ML Validator            │
│  • Backtesting UI                  │  • Risk Manager            │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
      ┌───────▼───────┐             ┌─────────▼─────────┐
      │  MetaTrader 5 │             │     MongoDB       │
      │   (Broker)    │             │   (Trade Data)    │
      └───────────────┘             └───────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **main.py** | Core trading bot with market monitoring, order execution, and position management |
| **app.py** | Flask REST API server for frontend communication |
| **strategies/** | Trading strategy implementations (SMC, Fibonacci, Liquidity Sweep, etc.) |
| **consensus.py** | Multi-strategy voting system for trade signal validation |
| **backtester.py** | Historical backtesting engine |
| **frontend/TradePilot/** | React dashboard for monitoring and control |

### Trading Strategies

- **SMC (Smart Money Concepts)** - Institutional order flow analysis
- **Fibonacci Retracement** - Key level identification
- **Liquidity Sweep** - Stop hunt detection
- **Malaysian SNR** - Support/Resistance breakout
- **ADX Strategy** - Trend strength filtering
- **ML Validation** - XGBoost model for signal confirmation

---

## 📋 Prerequisites

- **Python 3.10+** 
- **Node.js 18+** and npm
- **MetaTrader 5** terminal installed and configured
- **MongoDB** database (local or cloud - MongoDB Atlas)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "EA_ForexBot"
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Configure Environment Variables

Create a `.env` file in the root directory:

```env
# MongoDB Connection
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/<database>?retryWrites=true&w=majority
MONGO_DB_NAME=EA_ForexBot

# MT5 Credentials (optional - can be set via UI)
MT5_LOGIN_ID=your_login_id
MT5_SERVER=Broker-Server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

#### Configure Trading Parameters

Edit `config/config.json` to customize:
- Trading symbols (e.g., `XAUUSDm`, `BTCUSDm`)
- Timeframes (`M5`, `M15`)
- Risk management settings
- Strategy parameters
- Trading sessions

### 3. Frontend Setup

```bash
cd frontend/TradePilot
npm install
```

---

## 🏃 Running the Application

### Start the Backend (Flask API + Trading Bot)

```bash
# From the project root directory (with venv activated)
python app.py
```

The Flask server will start on `http://localhost:5000`

> **Note:** The trading bot does NOT start automatically. Use the Dashboard UI to start/stop the bot.

### Start the Frontend (React Dashboard)

```bash
# In a new terminal
cd frontend/TradePilot
npm run dev
```

The dashboard will be available at `http://localhost:5173`

---

## 📊 Using the Dashboard

| Page | Description |
|------|-------------|
| **Dashboard** | Real-time account balance, equity, open positions, and performance metrics |
| **Trades** | Complete trade history with filtering and sorting |
| **Analytics** | Strategy performance breakdown, win rates, and profit analysis |
| **Backtest** | Run historical backtests and analyze results |
| **Settings** | Configure MT5 credentials, symbols, risk parameters, and strategy settings |

### Starting the Trading Bot

1. Navigate to the Dashboard
2. Ensure MT5 is connected (green status indicator)
3. Click "Start Bot" to begin automated trading
4. Monitor trades and performance in real-time

---

## 📁 Project Structure

```
ForexBot - Experiments (Long Trades)/
├── app.py                 # Flask REST API server
├── main.py                # Core trading bot logic
├── backtester.py          # Backtesting engine
├── consensus.py           # Multi-strategy voting system
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
│
├── config/
│   └── config.json        # Trading configuration
│
├── strategies/
│   ├── smc.py             # Smart Money Concepts
│   ├── fibonacci.py       # Fibonacci retracement
│   ├── liquidity_sweep.py # Liquidity sweep detection
│   ├── malaysian_snr.py   # Support/Resistance
│   ├── adx_strategy.py    # ADX trend filter
│   └── ml_model.py        # ML validation (XGBoost)
│
├── utils/
│   ├── risk_manager.py    # Position sizing & risk
│   └── ...                # Other utilities
│
├── models/                # Trained ML models (.joblib)
├── logs/                  # Application logs
│
└── frontend/TradePilot/   # React dashboard
    ├── src/
    │   ├── pages/         # Dashboard, Trades, Analytics, Settings
    │   ├── components/    # Reusable UI components
    │   └── services/      # API communication
    └── package.json
```

---

## ⚙️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/account_info` | GET | Account balance, equity, weekly growth |
| `/api/bot/status` | GET | Bot running status |
| `/api/bot/start` | POST | Start the trading bot |
| `/api/bot/stop` | POST | Stop the trading bot |
| `/api/positions` | GET | Current open positions |
| `/api/trades` | GET | Trade history |
| `/api/strategy_performance` | GET | Performance by strategy |
| `/api/equity_curve` | GET | Equity curve data |
| `/api/config` | GET | Get configuration |
| `/api/config/update` | POST | Update configuration |

---

## 🔧 Troubleshooting

### MT5 Connection Failed
- Ensure MetaTrader 5 is installed and running
- Verify login credentials in Settings
- Check that the MT5 terminal path is correct

### MongoDB Connection Error
- Verify your `MONGO_URI` in `.env`
- Ensure your IP is whitelisted in MongoDB Atlas
- Check network connectivity

### Frontend Not Loading
- Ensure backend is running on port 5000
- Check for CORS issues in browser console
- Verify `npm install` completed successfully

---

## 📝 License

This project is for educational and personal use only. Trading involves significant risk of loss.

---

## ⚠️ Disclaimer

**RISK WARNING:** Forex and cryptocurrency trading carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results. Only trade with money you can afford to lose.
