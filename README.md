# AlphaSuite

AlphaSuite is an open-source quantitative analysis platform that gives you the power to build, test, and deploy professional-grade trading strategies. It's designed for traders and analysts who want to move beyond simple backtests and develop a genuine, data-driven edge in the financial markets.

## ✨ Key Features
 
*   **Modular Strategy Engine**: A powerful, `pybroker`-based engine for rigorous backtesting.
    *   **Walk-Forward Analysis**: Test strategies on out-of-sample data to prevent overfitting and ensure robustness.
    *   **Bayesian Optimization**: Automatically tune strategy parameters to find the most optimal settings.
    *   **ML Integration**: Seamlessly integrate machine learning models (like LightGBM) into your strategies.
    *   **Built-in Risk Management**: Enforces professional-grade position sizing and stop-loss mechanics based on volatility (e.g., ATR) and predefined risk-per-trade, a core feature often missing in other platforms.
    *   **Context-Aware Design**: Provides the framework and tools for strategies to implement dynamic market regime detection and adaptive logic, allowing models to learn and evolve with changing market conditions.
    *   **Extensible SDK**: Add new, complex trading strategies by creating a single Python file.
*   **Powerful Market Scanning**: A fully customizable scanner to find trading opportunities across global markets.
    *   **Generic Screener**: A rich UI to build custom screens using dozens of fundamental and technical filters without writing code.
    *   **Custom Scanner SDK**: An extensible framework to create scanners for any pattern imaginable, from RSI divergences to complex Wyckoff setups.
    *   **20+ Pre-Built Scanners**: Comes with a rich library of ready-to-use scanners for common trading patterns.
*   **Comprehensive Data & Research Tools**: An integrated suite for deep market analysis.
    *   **Automated Data Pipeline**: Fetches and stores comprehensive data for global markets from Yahoo Finance into a PostgreSQL database.
    *   **AI-Powered Stock Reports**: Generate in-depth fundamental and technical analysis reports for any stock using LLMs.
    *   **AI News Intelligence**: Get AI-generated market briefings and risk analysis based on the latest financial news.

## 🖼️ Screenshots

Here's a glimpse of what you can do with AlphaSuite.

### Home Page
![AlphaSuite Home Page](images/AlphaSuite_Home.jpg)

### Backtest Performance Visualization

Analyze the out-of-sample performance of a trained and tuned strategy model.

**Summary Metrics & Equity Curve:**
![Summary Metrics](images/SPY_donchian_breakout_summary_metrics.jpg)
![Equity Curve](images/SPY_donchian_breakout_equity_curve.jpg)

**Trade Execution Chart:**
![Trade Chart](images/SPY_donchian_breakout_trade_chart.jpg)

**Detailed Metrics Table:**
![Detailed Metrics](images/SPY_donchian_breakout_metrics.jpg)

## 💡 What Makes AlphaSuite Different?

In a world of numerous trading platforms, AlphaSuite was built to fill a specific gap: bridging the divide between simplistic backtesters and complex, institutional-grade systems. Its core philosophy is that modern trading strategies should be adaptive and data-driven from the ground up.

Unlike many platforms where machine learning is an optional add-on, AlphaSuite is **ML-native**. Its entire workflow is designed to be driven by ML models, reflecting the shift in quantitative finance away from static rules and toward dynamic strategies that learn from the market.

Key differentiators include:
*   **End-to-End ML-Centric Workflow**: Strategies are built around ML features, not just enhanced by them. The model's output *is* the signal.
*   **Built-in Professional Risk Management**: The framework's core trading logic enforces disciplined, non-negotiable risk management. Position sizing is systematically calculated based on volatility (e.g., ATR) and a predefined risk-per-trade, a feature often missing from traditional backtesters. This enforces capital preservation by design.
*   **Enables Strategy-Driven Context Awareness**: While not a built-in feature, the framework provides the tools and hooks for strategies to become *regime-aware*. Developers can easily incorporate market context (like volatility regimes or trend strength) into their features, allowing models to learn *when* and *why* a setup is effective.
*   **Focus on Robust Validation**: With tools like walk-forward analysis and portfolio-level backtesting, AlphaSuite emphasizes proving a strategy's edge on unseen data, avoiding the common pitfalls of overfitting.

In short, AlphaSuite isn’t just about asking “would this have made money?”—it’s about building strategies that understand *when* to trade, *how much* to risk, and *why* they have an edge.

## 📖 Articles & Case Studies

Check out these articles to see how AlphaSuite can be used to develop and test sophisticated trading strategies from scratch:

*   **[Stop Paying for Stock Screeners. Build Your Own for Free with Python](https://medium.com/codex/stop-paying-for-stock-screeners-build-your-own-for-free-with-python-222b52d324b5)**: A comprehensive guide on using AlphaSuite's modular market scanner to build custom screens for any market, moving beyond the limitations of commercial tools.
*   **[From Backtest to Battle-Ready: A Guide to Preparing a Trading Strategy with AlphaSuite](https://medium.com/codex/from-backtest-to-battle-ready-a-guide-to-preparing-a-trading-strategy-with-alphasuite-23d765085cd6)**: A practical, step-by-step walkthrough for taking a strategy from concept to live-trading readiness using our open-source quant engine.
*   **[We Backtested a Viral Trading Strategy. The Results Will Teach You a Lesson.](https://medium.com/codex/we-backtested-a-viral-trading-strategy-the-results-will-teach-you-a-lesson-b57d7c9bfb74)**: An investigation into a popular trading strategy, highlighting critical lessons on overfitting, data leakage, and the importance of robust backtesting. Also available as a [video narration](https://youtu.be/-cKYPx43jTg).
*   **[I Was Paralyzed by Uncertainty, So I Built My Own Quant Engine](https://medium.com/codex/i-was-paralyzed-by-stock-market-uncertainty-so-i-built-my-own-quant-engine-176a6706c451)**: The story behind AlphaSuite's creation and its mission to empower data-driven investors. Also available as a [video narration](https://youtu.be/NXk7bXPYGP8).
*   **[From Chaos Theory to a Profitable Trading Strategy](https://medium.com/codex/from-chaos-theory-to-a-profitable-trading-strategy-in-30-minutes-d247cba4bbbd)**: A step-by-step guide on building a rule-based strategy using concepts from chaos theory.
*   **[Supercharging a Machine Learning Strategy with Lorenz Features](https://medium.com/codex/from-chaos-to-alpha-part-2-supercharging-a-machine-learning-strategy-with-lorenz-features-794acfd3f88c)**: Demonstrates how to enhance an ML-based strategy with custom features and optimize it using walk-forward analysis.
*   **[The Institutional Edge: How We Boosted a Strategy’s Return with Volume Profile](https://medium.com/codex/the-institutional-edge-how-we-boosted-a-strategys-return-from-162-to-223-with-one-indicator-eef74cadae91)**: A deep dive into using Volume Profile to enhance a classic trend-following strategy, demonstrating a significant performance boost.
*   **[Trading with Less Surprise: Using Shannon Entropy to Improve a Breakout Strategy](https://medium.com/codex/trading-with-less-surprise-using-shannon-entropy-to-improve-a-breakout-strategy-4d0a15098cba)**: Demonstrates how a concept from information theory can dramatically improve a classic trend-following strategy. 
*   **[The Feature Engineering Treadmill: Why One “Magic” Feature Isn’t Enough](https://medium.com/codex/the-feature-engineering-treadmill-why-one-magic-feature-isnt-enough-fd2ac4f396e2)**: Explores how to evolve a promising idea into a more robust, multi-asset system.
*   **[Your Trading Strategy Has No Context and That’s the Problem](https://medium.com/codex/ai-can-understand-context-can-your-trading-strategy-bfacdf6d66a1)**: Explores the importance of market regime features, compare several methods we’ve tested using AlphaSuite.
*   **[Beyond the Almanac: Supercharging Trading Strategies with On-Demand Seasonality Analysis](https://medium.com/codex/beyond-the-almanac-supercharging-trading-strategies-with-on-demand-seasonality-analysis-063bf7d58f2c)**: Introudces AlphaSuite's new on-demand seasonality analysis feature that can analyze seasonality on the fly for a specific ticker.
*   **[Algo Trading Series, Part 1: The Resolution — Stop Guessing, Start Measuring](https://medium.com/codex/algo-trading-series-part-1-the-resolution-stop-guessing-start-measuring-330509a28d5d)**: Why your New Year’s resolution shouldn’t be “trade better” — it should be “trade differently.”
*   **[Algo Trading Series, Part 2: The AI Toolkit — Architecture of a Quant Engine](https://medium.com/codex/algo-trading-series-part-2-the-ai-toolkit-architecture-of-a-quant-engine-fc03547f95b7)**: How machines really learn to trade, and why a robust testing framework is non-negotiable.
*   **[Algo Trading Series, Part 3: The Concept — Hunting for Liquidity](https://medium.com/codex/algo-trading-series-part-3-the-concept-hunting-for-liquidity-a689a16b807b)**: Why “Smart Money” hunts stops, and how to write Python code to find them.
*   **[Algo Trading Series, Part 4: Codifying the Edge — When Textbooks Meet Reality](https://medium.com/codex/algo-trading-series-part-4-codifying-the-edge-when-textbooks-meet-reality-e7377f51282b)**: How to build a ML strategy based on the “Smart Money Concepts.”
*   **[Algo Trading Series, Part 5: The Optimization — From Theory to Production](https://medium.com/codex/algo-trading-series-part-5-the-optimization-from-theory-to-production-70421f88ee9c)**: How to evolve the structure_liquidity strategy into an adaptive, Machine Learning-driven sniper.
*   **[Algo Trading Series, Part 6: The Litmus Test — If You Feed It Garbage, Does It Still Eat?](https://medium.com/codex/algo-trading-series-part-6-the-litmus-test-if-you-feed-it-garbage-does-it-still-eat-6d2a29874c89)**: We stress-test the strategy with Noise Injection to ensure the edge is real and not just overfitting.

## 🛠️ Tech Stack

*   **Backend**: Python
*   **Web Framework**: Streamlit
*   **Backtesting Engine**: [pybroker](https://github.com/edtechre/pybroker)
*   **Data Analysis**: Pandas, NumPy, SciPy
*   **Financial Data**: yfinance, EODHD Financial Data API, TA-Lib
*   **Database**: PostgreSQL with SQLAlchemy
*   **AI/LLM**: LangChain, Google Gemini, Ollama

## 🔌 Data Provider Architecture & Extended Support

AlphaSuite now features a modular, provider-agnostic data integration pipeline. Historically built on `yfinance`, the platform has been refactored around a unified `DataProviderFactory` pattern and standardized SQLAlchemy entities (`Company`, `Financials`, `AnalystEstimates`, `InsiderRoster`, `Holdings`, `UpgradeDowngrade`). 

This architecture decoupling allows developers to plug in custom or commercial data sources without altering downstream strategy logic, backtest engines, or database models.

### Supported Providers Out-of-the-Box:
1. **`yfinance` (Default)**: Free, open-source integration for quick setups and basic market backtesting.
2. **[EODHD APIs](https://eodhd.com/)** *(Recommended)*: Institutional-grade global coverage including real-time, historical, and end-of-day prices, extended fundamentals with analyst sentiment/estimates, insider rosters, and institutional holdings, etc.

> **Get Started with Institutional Data:**
> To enable institutional-grade historical data and deep fundamental coverage, sign up via our partner link: **[EODHD APIs](https://eodhd.com/?via=alpha_suite&utm_source=github&utm_medium=lib&utm_campaign=alpha_suite&utm_content=richard_shu)** and configure your `EODHD_API_KEY` in `.env`.

## 📂 Project Structure

The project is organized into several key directories:

*   `core/`: Contains the core application logic, including database setup (`db.py`), model definitions (`model.py`), data providers, and logging configuration.
*   `pages/`: Each file in this directory corresponds to a page in the Streamlit web UI.
*   `pybroker_trainer/`: Holds the machine learning pipeline for training and tuning trading models with `pybroker`.
*   `strategies/`: Contains the definitions for different trading strategies. New strategies can be added here.
*   `scanners/`: Contains the definitions for custom market scanners. New scanners can be added here.
*   `tools/`: Includes various utility modules for tasks like financial calculations, data scanning, and interacting with the financial data providers API.
*   `Home.py`: The main entry point for the Streamlit application.
*   `download_data.py`: The command-line interface for all data management tasks.
*   `quant_engine.py`: The core quantitative engine for backtesting and analysis.
*   `requirements.txt`: A list of all the Python packages required to run the project.
*   `.env.example`: An example file for environment variables.

## 🚀 Getting Started

Follow these steps to set up and run AlphaSuite on your local machine.

### 1. Prerequisites

*   Python 3.9+
*   PostgreSQL Server
*   Git

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rsandx/AlphaSuite.git
    cd AlphaSuite
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    *   **TA-Lib**: This library has a C dependency that must be installed first. Follow the official [TA-Lib installation instructions](https://github.com/mrjbq7/ta-lib) for your operating system.
    *   Install the remaining Python packages:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Set up the Database:**
    *   Ensure your PostgreSQL server is running.
    *   Create a new database (e.g., `alphasuite`).
    *   The application will create the necessary tables on its first run.

5.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and edit the variables:
        *   `DATABASE_URL`: Set this to your PostgreSQL connection string.
        *   `LLM_PROVIDER`: Set to `gemini` or `ollama` to choose your provider.
        *   `GEMINI_API_KEY`: Required if `LLM_PROVIDER` is `gemini`.
        *   `OLLAMA_URL`: The URL for your running Ollama instance (e.g., `http://localhost:11434`). Required for `ollama`.
        *   `OLLAMA_MODEL`: The name of the model you have pulled in Ollama (e.g., `llama3`).

### 3. Usage

1.  **Initial Data Download:**
    Before running the app, you need to populate the database with market data. Run the download script from your terminal. This may take a long time for the initial run.
    *   For the **very first run** to populate your database:
       To Initialize dababase 
        ```bash
		python download_data.py init-db
        ```
        To validate dababase 
        ```bash
        python download_data.py scan
        ```
        To download and populate database
        ```bash
        python download_data.py download 
        ```
    ```bash
    # For subsequent daily updates, run the pipeline:
    python download_data.py pipeline
    ```

2.  **Run the Streamlit Web Application:**
    ```bash
    streamlit run Home.py
    ```
    Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Follow the In-App Workflow:**
    1.  **Populate Data:** Go to the **Data Management** page and run the "Daily Pipeline" or a "Full Download".
    2.  **Scan for Setups:** Go to the **Market Scanner** page. Use the "Signal Scanner" to find setups from your trained ML models, or use the "Generic Scanner" to find stocks matching fundamental or technical criteria with custom scanners.
    3.  **Tune & Train:** To build custom models, navigate to the **Model Training & Tuning** page.
    4.  **Analyze & Backtest:** Use the **Portfolio Analysis** page to validate your strategies.
    5.  **Deep Research:** Use the **Stock Report** page for in-depth analysis of specific stocks.

## 🧠 Adding a New Trading Strategy

The quantitative engine is designed to be modular, allowing new trading strategies to be developed and integrated by simply adding a single, self-contained Python file to the `strategies/` directory. The system automatically discovers and loads any valid strategy file at runtime.

### How It Works

The system scans the `strategies/` directory for Python files. Inside each file, it looks for a class that inherits from `pybroker_trainer.strategy_sdk.BaseStrategy`. This class encapsulates all the logic and parameters for a single strategy. Two sample strategy files are included in the repository:
ma_crossover.py is a non-ML, rule-based strategy; donchian_breakout.py is a ML-based strategy. You can use them as a reference for building your own.

### Step-by-Step Guide

1.  **Create a New File:** Create a new Python file in the `strategies/` directory. The filename should be descriptive and use snake_case (e.g., `my_awesome_strategy.py`).
2.  **Define the Strategy Class:** Inside the new file, define a class that inherits from `BaseStrategy`. The class name should be descriptive and use CamelCase (e.g., `MyAwesomeStrategy`).
3.  **Implement Required Methods:** Implement the four required methods within your class: `define_parameters`, `get_feature_list`, `add_strategy_specific_features`, and `get_setup_mask`.

### Strategy Class Breakdown

Each strategy class must implement the following methods, which define its behavior, data requirements, and entry logic.

#### 1. `define_parameters()`

This static method defines all the parameters the strategy uses, their default values, and their tuning ranges for optimization. This is critical for backtesting and hyperparameter tuning.

*   **Returns:** A dictionary where each key is a parameter name. The value is another dictionary specifying its `type`, `default` value, and a `tuning_range` tuple.

**Example from `DonchianBreakoutStrategy`:**
```python
@staticmethod
def define_parameters():
    """Defines parameters, their types, defaults, and tuning ranges."""
    return {
        'donchian_period': {'type': 'int', 'default': 20, 'tuning_range': (15, 50)},
        'atr_period': {'type': 'int', 'default': 14, 'tuning_range': (10, 30)},
        # ... other parameters
    }
```

#### 2. `get_feature_list()`

This method returns a list of all the feature (column) names that the strategy's machine learning model requires as input. The training engine uses this list to prepare the data correctly.

*   **Returns:** A `list` of strings.

**Example from `DonchianBreakoutStrategy`:**
```python
def get_feature_list(self) -> list[str]:
    """Returns the list of feature column names required by the model."""
    return [
        'roc', 'rsi', 'mom', 'ppo', 'cci',
        # ... other features
    ]
```

#### 3. `add_strategy_specific_features()`

This is where you calculate any indicators or features that are unique to your strategy and are not part of the common features provided by the system.

*   **Arguments:** A pandas `DataFrame` containing the price data and common indicators.
*   **Returns:** The modified pandas `DataFrame` with your new feature columns added.

**Example from `DonchianBreakoutStrategy`:**
```python
def add_strategy_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds features unique to this specific strategy."""
    donchian_period = self.params.get('donchian_period', 20)
    data['donchian_upper'] = data['high'].rolling(window=donchian_period).max()
    data['donchian_lower'] = data['low'].rolling(window=donchian_period).min()
    data['donchian_middle'] = (data['donchian_upper'] + data['donchian_lower']) / 2
    return data
```

#### 4. `get_setup_mask()`

This is the core of your strategy's entry logic. This method must return a boolean pandas `Series` that is `True` on the bars where a potential trade setup occurs and `False` otherwise.

*   **Arguments:** A pandas `DataFrame` containing all required features (both common and strategy-specific).
*   **Returns:** A pandas `Series` of boolean values, with the same index as the input `DataFrame`.

**Example from `DonchianBreakoutStrategy`:**
```python
def get_setup_mask(self, data: pd.DataFrame) -> pd.Series:
    """Returns a boolean Series indicating the bars where a trade setup occurs."""
    is_uptrend = data['trend_bullish'] == 1
    is_breakout = data['high'] > data['donchian_upper'].shift(1)
    raw_setup_mask = is_uptrend & is_breakout
    # Ensure we only signal on the first bar of a new setup
    return raw_setup_mask & ~raw_setup_mask.shift(1).fillna(False)
```

By following this structure, you can create new, complex strategies that seamlessly integrate with the project's backtesting, tuning, and training infrastructure.

## 🧠 Adding a New Scanner

AlphaSuite comes with over 20 pre-built custom scanners that are ready to use out of the box. These scanners cover a wide range of technical and fundamental patterns, from classic RSI divergences to complex Wyckoff accumulation setups. They not only provide powerful screening capabilities but also serve as excellent, practical examples for developers looking to build their own custom scanners.

The Market Scanner is also designed to be modular, allowing you to create and integrate custom scanners with minimal effort. By adding a self-contained Python file to the `scanners/` directory, the system will automatically discover and load it into the Streamlit UI.

### How It Works

The system scans the `scanners/` directory for Python files. Inside each file, it looks for a class that inherits from `scanners.scanner_sdk.BaseScanner`. This class encapsulates all the logic and parameters for a single scanner. The `generic_screener.py` is a special, built-in scanner, but any other file you add will be treated as a custom scanner.

### Step-by-Step Guide

1.  **Create a New File:** Create a new Python file in the `scanners/` directory. The filename should be descriptive and use snake_case (e.g., `my_custom_scanner.py`). The filename will be used as the scanner's unique identifier.
2.  **Define the Scanner Class:** Inside the new file, define a class that inherits from `BaseScanner`. The class name should be descriptive and use CamelCase (e.g., `MyCustomScanner`).
3.  **Implement Required Methods:** Implement the required methods within your class to define its parameters, display columns, and scanning logic.

### Scanner Class Breakdown

Each scanner class should implement the following methods to define its behavior.

#### 1. `define_parameters()`

This static method defines the parameters that will appear in the UI for your scanner. It allows users to customize the scan without changing the code.

*   **Returns:** A `list` of dictionaries, where each dictionary defines a parameter's `name`, `type` (`int`, `float`, `select`), `default` value, and `label`.

**Example from `BullishDipBounceScanner`:**
```python
@staticmethod
def define_parameters():
    return [
        {"name": "min_avg_volume", "type": "int", "default": 250000, "label": "Min. Avg. Volume"},
        {"name": "rsi_period", "type": "int", "default": 7, "label": "RSI Period"},
        {"name": "divergence_lookback", "type": "int", "default": 30, "label": "Divergence Lookback"},
    ]
```

#### 2. `get_leading_columns()`

This static method returns a list of column names that should be displayed first in the results table, ensuring the most important information is easily visible.

*   **Returns:** A `list` of strings.

**Example from `BullishDipBounceScanner`:**
```python
@staticmethod
def get_leading_columns():
    return ['symbol', 'rsi', 'divergence_date', 'longname', 'marketcap']
```

#### 3. `get_sort_info()`

This static method defines the default sorting order for the results table.

*   **Returns:** A `dict` specifying the column(s) to sort `by` and the `ascending` order.

**Example from `BullishDipBounceScanner`:**
```python
@staticmethod
def get_sort_info():
    return {'by': 'marketcap', 'ascending': False}
```

#### 4. `scan_company()`

This is the core logic of your scanner. The base framework handles filtering stocks by market, volume, and market cap, then calls this method for each remaining company. Your job is to analyze the provided data and determine if the company is a match.

*   **Arguments:**
    *   `group`: A pandas `DataFrame` containing the company's historical price data.
    *   `company_info`: A `dict` containing basic company information from the database.
*   **Returns:** The `company_info` dictionary if the stock matches the criteria (you can add new keys to it, like `divergence_date`), or `None` if it does not.

**Example from `BullishDipBounceScanner`:**
```python
def scan_company(self, group: pd.DataFrame, company_info: dict) -> dict | None:
    # ... (calculation logic for RSI, SMA, and divergence) ...

    if is_uptrend and is_lower_low_price and is_higher_rsi:
        company_info['rsi'] = float(rsi.iloc[-i])
        company_info['divergence_date'] = group['date'].iloc[-i].strftime('%Y-%m-%d')
        return company_info
    
    return None
```

For scanners with more complex requirements that don't fit the per-company iteration model (like the `StrongestIndustriesScanner`, which groups by industry first), you can override the `run_scan()` method to implement your own custom data fetching and processing logic.

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.
