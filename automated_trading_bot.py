
"""
Automated Trading System with CloudAI + Zerodha MCP Integration
Free Setup for Bank Nifty and Nifty 50 Analysis

Author: Trading Assistant
Version: 1.0
Dependencies: See requirements.txt
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
import requests
import smtplib
import json
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import telegram
from telegram import Bot
import warnings

class CloudAIIntegration:
    def __init__(self, config):
        self.endpoint = config.get('cloudai_endpoint')
        self.api_key = config.get('zerodha_api_key')

    def get_live_positions(self):
        """Get current positions from Zerodha via CloudAI"""
        pass

    def get_live_market_data(self):
        """Get real-time market data via CloudAI"""
        pass

    def analyze_portfolio_context(self):
        """Get portfolio-specific insights via CloudAI"""
        pass
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    # CloudAI + MCP Integration
    "cloudai_endpoint": "your_cloudai_mcp_endpoint",
    "zerodha_api_key": "your_zerodha_api_key",
    "zerodha_access_token": "your_zerodha_access_token",

    # Notification Settings
    "telegram_bot_token": "your_telegram_bot_token",
    "telegram_chat_id": "your_telegram_chat_id",
    "email_user": "your_email@gmail.com",
    "email_password": "your_app_password",
    "notification_email": "recipient@gmail.com",

    # Market Settings
    "symbols": ["^NSEI", "^NSEBANK"],  # Nifty 50, Bank Nifty
    "timeframe": "1d",
    "lookback_days": 100,
    "support_resistance_periods": 20,
    "volume_threshold": 1.5,  # 50% above average

    # Analysis Settings
    "rsi_period": 14,
    "ma_short": 20,
    "ma_long": 50,
    "bollinger_period": 20,
    "volume_ma_period": 20
}

class MarketDataFetcher:
    """Fetch market data from multiple free sources"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_nse_data(self, symbol):
        """Get NSE data using free APIs"""
        try:
            # Primary: Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="100d", interval="1d")
            return data
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def get_option_chain(self, symbol="NIFTY"):
        """Get option chain data (free method)"""
        try:
            # Note: This is a simplified version
            # In practice, you might need to scrape NSE website or use unofficial APIs
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error fetching option chain for {symbol}: {e}")
            return None

    def get_news_sentiment(self, query="nifty bank nifty"):
        """Get news sentiment using free APIs"""
        try:
            # Using NewsAPI free tier (100 requests/day)
            api_key = "your_newsapi_key"  # Get free from newsapi.org
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={api_key}"
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error fetching news: {e}")
            return None

class TechnicalAnalyzer:
    """Perform technical analysis on market data"""

    def __init__(self, data):
        self.data = data
        self.close = data['Close'].values
        self.high = data['High'].values
        self.low = data['Low'].values
        self.volume = data['Volume'].values

    def calculate_support_resistance(self, window=20):
        """Calculate dynamic support and resistance levels"""
        highs = self.data['High'].rolling(window=window).max()
        lows = self.data['Low'].rolling(window=window).min()

        # Find significant levels using pivot points
        resistance_levels = []
        support_levels = []

        for i in range(window, len(self.data)):
            current_high = self.data['High'].iloc[i]
            current_low = self.data['Low'].iloc[i]

            # Check if current high is a resistance
            if (current_high == highs.iloc[i] and 
                current_high > self.data['High'].iloc[i-1] and
                current_high > self.data['High'].iloc[i+1] if i < len(self.data)-1 else True):
                resistance_levels.append(current_high)

            # Check if current low is a support
            if (current_low == lows.iloc[i] and 
                current_low < self.data['Low'].iloc[i-1] and
                current_low < self.data['Low'].iloc[i+1] if i < len(self.data)-1 else True):
                support_levels.append(current_low)

        return {
            'resistance': sorted(resistance_levels, reverse=True)[:3],  # Top 3
            'support': sorted(support_levels)[-3:]  # Bottom 3
        }

    def calculate_indicators(self):
        """Calculate various technical indicators"""
        indicators = {}

        # Price-based indicators
        indicators['rsi'] = talib.RSI(self.close, timeperiod=14)
        indicators['sma_20'] = talib.SMA(self.close, timeperiod=20)
        indicators['sma_50'] = talib.SMA(self.close, timeperiod=50)
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(self.close)
        indicators['macd'], indicators['macd_signal'], _ = talib.MACD(self.close)

        # Volume indicators
        indicators['obv'] = talib.OBV(self.close, self.volume)
        indicators['ad'] = talib.AD(self.high, self.low, self.close, self.volume)
        indicators['mfi'] = talib.MFI(self.high, self.low, self.close, self.volume)

        return indicators

    def detect_volume_breakout(self, threshold=1.5):
        """Detect volume breakouts"""
        volume_ma = talib.SMA(self.volume, timeperiod=20)
        current_volume = self.volume[-1]
        avg_volume = volume_ma[-1]

        return {
            'is_breakout': current_volume > (avg_volume * threshold),
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 0,
            'current_volume': current_volume,
            'avg_volume': avg_volume
        }

    def analyze_price_action(self):
        """Analyze current price action"""
        current_price = self.close[-1]
        prev_price = self.close[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100

        # Get support/resistance
        sr_levels = self.calculate_support_resistance()

        # Check proximity to S/R levels
        sr_analysis = {
            'near_resistance': False,
            'near_support': False,
            'closest_resistance': None,
            'closest_support': None
        }

        for resistance in sr_levels['resistance']:
            if abs(current_price - resistance) / current_price < 0.02:  # Within 2%
                sr_analysis['near_resistance'] = True
                sr_analysis['closest_resistance'] = resistance
                break

        for support in sr_levels['support']:
            if abs(current_price - support) / current_price < 0.02:  # Within 2%
                sr_analysis['near_support'] = True
                sr_analysis['closest_support'] = support
                break

        return {
            'price_change_percent': price_change,
            'current_price': current_price,
            'sr_analysis': sr_analysis,
            'sr_levels': sr_levels
        }

class OpportunityDetector:
    """Detect trading opportunities based on multiple factors"""

    def __init__(self, analyzer, volume_analysis, news_sentiment=None):
        self.analyzer = analyzer
        self.volume_analysis = volume_analysis
        self.news_sentiment = news_sentiment
        self.indicators = analyzer.calculate_indicators()
        self.price_action = analyzer.analyze_price_action()

    def generate_signals(self):
        """Generate trading signals based on confluence"""
        signals = []

        current_rsi = self.indicators['rsi'][-1]
        current_price = self.price_action['current_price']
        price_change = self.price_action['price_change_percent']

        # Bullish signals
        bullish_score = 0
        if current_rsi < 30:  # Oversold
            bullish_score += 2
        if self.price_action['sr_analysis']['near_support']:  # Near support
            bullish_score += 2
        if self.volume_analysis['is_breakout'] and price_change > 0:  # Volume breakout up
            bullish_score += 3
        if self.indicators['macd'][-1] > self.indicators['macd_signal'][-1]:  # MACD bullish
            bullish_score += 1

        # Bearish signals
        bearish_score = 0
        if current_rsi > 70:  # Overbought
            bearish_score += 2
        if self.price_action['sr_analysis']['near_resistance']:  # Near resistance
            bearish_score += 2
        if self.volume_analysis['is_breakout'] and price_change < 0:  # Volume breakout down
            bearish_score += 3
        if self.indicators['macd'][-1] < self.indicators['macd_signal'][-1]:  # MACD bearish
            bearish_score += 1

        # Generate signals based on scores
        if bullish_score >= 4:
            signals.append({
                'type': 'BUY',
                'strength': 'STRONG' if bullish_score >= 6 else 'MODERATE',
                'score': bullish_score,
                'reasons': self._get_signal_reasons('bullish')
            })

        if bearish_score >= 4:
            signals.append({
                'type': 'SELL',
                'strength': 'STRONG' if bearish_score >= 6 else 'MODERATE',
                'score': bearish_score,
                'reasons': self._get_signal_reasons('bearish')
            })

        return signals

    def _get_signal_reasons(self, signal_type):
        """Get detailed reasons for the signal"""
        reasons = []

        if signal_type == 'bullish':
            if self.indicators['rsi'][-1] < 30:
                reasons.append("RSI oversold")
            if self.price_action['sr_analysis']['near_support']:
                reasons.append("Price near support level")
            if self.volume_analysis['is_breakout']:
                reasons.append(f"Volume breakout ({self.volume_analysis['volume_ratio']:.2f}x avg)")

        elif signal_type == 'bearish':
            if self.indicators['rsi'][-1] > 70:
                reasons.append("RSI overbought")
            if self.price_action['sr_analysis']['near_resistance']:
                reasons.append("Price near resistance level")
            if self.volume_analysis['is_breakout']:
                reasons.append(f"Volume breakout ({self.volume_analysis['volume_ratio']:.2f}x avg)")

        return reasons

class NotificationManager:
    """Handle various notification methods"""

    def __init__(self, config):
        self.config = config
        self.telegram_bot = None
        if config.get('telegram_bot_token'):
            self.telegram_bot = Bot(token=config['telegram_bot_token'])

    def send_telegram(self, message):
        """Send Telegram notification"""
        try:
            if self.telegram_bot and self.config.get('telegram_chat_id'):
                self.telegram_bot.send_message(
                    chat_id=self.config['telegram_chat_id'],
                    text=message,
                    parse_mode='Markdown'
                )
                return True
        except Exception as e:
            print(f"Telegram error: {e}")
        return False

    def send_email(self, subject, body):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email_user']
            msg['To'] = self.config['notification_email']
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.config['email_user'], self.config['email_password'])
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Email error: {e}")
        return False

    def notify_opportunity(self, symbol, signals, analysis):
        """Send comprehensive opportunity notification"""
        message = f"""
🚨 *Trading Opportunity Alert* 🚨

📈 *Symbol*: {symbol}
💰 *Current Price*: ₹{analysis['current_price']:.2f}
📊 *Price Change*: {analysis['price_change_percent']:.2f}%

*Signals*:
"""

        for signal in signals:
            message += f"\n🔹 *{signal['type']}* ({signal['strength']})"
            message += f"\n   Score: {signal['score']}/7"
            message += f"\n   Reasons: {', '.join(signal['reasons'])}"

        message += f"""

*Support Levels*: {', '.join([f"₹{s:.2f}" for s in analysis['sr_levels']['support']])}
*Resistance Levels*: {', '.join([f"₹{r:.2f}" for r in analysis['sr_levels']['resistance']])}

⏰ *Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # Send via multiple channels
        self.send_telegram(message)
        self.send_email(
            f"Trading Alert: {symbol}",
            message.replace('*', '').replace('\n', '\n')
        )

class TradingBot:
    """Main trading bot orchestrator"""

    def __init__(self, config):
        self.config = config
        self.data_fetcher = MarketDataFetcher()
        self.notification_manager = NotificationManager(config)
        self.last_signals = {}  # Prevent duplicate alerts

    def run_analysis(self):
        """Run complete analysis for all symbols"""
        for symbol in self.config['symbols']:
            try:
                print(f"Analyzing {symbol}...")

                # Fetch data
                data = self.data_fetcher.get_nse_data(symbol)
                if data is None or len(data) < 50:
                    continue

                # Perform analysis
                analyzer = TechnicalAnalyzer(data)
                volume_analysis = analyzer.detect_volume_breakout()

                # Get news sentiment (optional)
                news_sentiment = self.data_fetcher.get_news_sentiment()

                # Detect opportunities
                detector = OpportunityDetector(analyzer, volume_analysis, news_sentiment)
                signals = detector.generate_signals()

                # Check if we should notify
                if signals and self._should_notify(symbol, signals):
                    price_action = analyzer.analyze_price_action()
                    self.notification_manager.notify_opportunity(symbol, signals, price_action)
                    self.last_signals[symbol] = {
                        'signals': signals,
                        'timestamp': datetime.now()
                    }

            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

    def _should_notify(self, symbol, signals):
        """Check if we should send notification to avoid spam"""
        if symbol not in self.last_signals:
            return True

        # Don't send same signal within 1 hour
        last_time = self.last_signals[symbol]['timestamp']
        if datetime.now() - last_time < timedelta(hours=1):
            return False

        return True

    def run_continuously(self, interval_minutes=5):
        """Run bot continuously"""
        print(f"Starting trading bot... Running every {interval_minutes} minutes")

        while True:
            try:
                self.run_analysis()
                print(f"Analysis complete. Next run in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("Bot stopped by user")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def main():
    """Main function to run the trading bot"""
    # Load configuration (you can modify this to load from file)
    config = CONFIG.copy()

    # Validate configuration
    required_fields = ['telegram_bot_token', 'telegram_chat_id']
    for field in required_fields:
        if not config.get(field):
            print(f"Warning: {field} not configured. Some features may not work.")

    # Create and run bot
    bot = TradingBot(config)

    # Run once for testing
    print("Running single analysis...")
    bot.run_analysis()

    # Uncomment to run continuously
    # bot.run_continuously(interval_minutes=5)

if __name__ == "__main__":
    main()
