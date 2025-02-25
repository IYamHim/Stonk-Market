import yfinance as yf
import requests
import pandas as pd
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import datetime
import os
import json
import time

# Try importing newsapi, but make it optional
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    print("NewsAPI Python client not installed. To use NewsAPI, install with:")
    print("pip install newsapi-python")
    NEWSAPI_AVAILABLE = False

# Configure your NewsAPI key
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"  # Replace with your actual key

class StockSearchEngine:
    def __init__(self, model_path="./qwen_stonk_advisor_final"):
        print("Initializing Stock Search Engine...")
        
        # Check if this is a base model or a fine-tuned model
        is_base_model = model_path.startswith("Qwen/")
        
        # Adjust paths for models in the train directory
        if not is_base_model and not os.path.exists(model_path):
            # Try looking in the train subdirectory
            train_path = os.path.join("train", os.path.basename(model_path))
            if os.path.exists(train_path):
                print(f"Model not found at {model_path}, using {train_path} instead")
                model_path = train_path
        
        # Load the model
        print("Loading model from:", model_path)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path if not is_base_model else model_path,
                trust_remote_code=True
            )
            
            if hasattr(self, 'model') and self.model is not None:
                # Model already loaded (used by DirectStockSearchEngine)
                print("Using pre-loaded model.")
            elif is_base_model:
                # Load base model directly
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.model = self.base_model
            else:
                # Load as PEFT model
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Check if adapter_config.json exists
                if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                    self.model = PeftModel.from_pretrained(self.base_model, model_path)
                else:
                    print(f"Warning: adapter_config.json not found in {model_path}")
                    print("Using base model without fine-tuning.")
                    self.model = self.base_model
                    
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to base model...")
            
            # Load base model as fallback
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = self.base_model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                trust_remote_code=True
            )
        
        # Initialize news API client
        self.news_client = None
        if NEWSAPI_AVAILABLE and NEWSAPI_KEY != "YOUR_NEWSAPI_KEY":
            try:
                self.news_client = NewsApiClient(api_key=NEWSAPI_KEY)
                print("NewsAPI client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize NewsAPI client: {str(e)}")
        else:
            print("Warning: NewsAPI not available. Using fallback news sources.")
            
        print("Stock Search Engine initialized successfully!")
    
    def get_stock_data(self, ticker):
        """Retrieve stock data from Yahoo Finance"""
        try:
            print(f"Fetching data for {ticker}...")
            # Get stock info
            stock = yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            if not info:
                print(f"No information found for ticker: {ticker}")
                return None
                
            # Get historical data for 5 days
            hist = stock.history(period="5d")
            
            # Format the data
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2]
            price_change_percent = ((current_price - previous_price) / previous_price) * 100
            
            # Get financial ratios
            financial_data = {}
            for key in ['trailingPE', 'forwardPE', 'marketCap', 'priceToBook', 'beta']:
                if key in info:
                    financial_data[key] = info[key]
            
            return {
                'ticker': ticker,
                'company_name': info.get('shortName', info.get('longName', 'Unknown')),
                'current_date': current_date,
                'current_price': current_price,
                'previous_price': previous_price,
                'price_change_percent': price_change_percent,
                'financials': financial_data,
                'description': info.get('longBusinessSummary', 'No description available')
            }
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return None
    
    def get_news(self, ticker, company_name):
        """Get recent news articles about the stock"""
        try:
            news_list = []
            
            if self.news_client:
                # Use NewsAPI if configured
                news = self.news_client.get_everything(
                    q=f"{ticker} OR {company_name}",
                    language='en',
                    sort_by='publishedAt',
                    page_size=5
                )
                
                if news['status'] == 'ok':
                    for article in news['articles']:
                        news_list.append({
                            'headline': article['title'],
                            'date': article['publishedAt'],
                            'source': article['source']['name']
                        })
            
            # Fallback to Yahoo Finance news if NewsAPI failed or not configured
            if not news_list:
                print("Using fallback news source...")
                url = f"https://finance.yahoo.com/quote/{ticker}/news"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    news_items = soup.find_all('h3', {'class': 'Mb(5px)'})
                    
                    for item in news_items[:5]:
                        news_list.append({
                            'headline': item.text,
                            'date': 'Recent',
                            'source': 'Yahoo Finance'
                        })
            
            return news_list
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return [{'headline': 'No recent news available', 'date': '', 'source': ''}]
    
    def format_prompt(self, stock_data, news):
        """Format the data into a prompt for the model"""
        financials_str = "\n".join([f"{k}: {v}" for k, v in stock_data['financials'].items()])
        news_str = "\n".join([f"- {item['headline']} ({item['source']})" for item in news])
        
        prompt = f"""Stock: {stock_data['ticker']}
Date: {stock_data['current_date']}
Company: {stock_data['company_name']}
Description: {stock_data['description'][:300]}...
Current Price: ${stock_data['current_price']:.2f}
Previous Close: ${stock_data['previous_price']:.2f}
Price Change: {stock_data['price_change_percent']:.2f}%

Recent News:
{news_str}

Financial Data:
{financials_str}

Question: Based on this information, analyze whether this stock will go up or down in the next trading day. Provide your reasoning and a specific prediction with a percentage range.
"""
        return prompt
    
    def predict(self, ticker):
        """Generate a prediction for the given stock ticker"""
        # Get stock data
        stock_data = self.get_stock_data(ticker)
        if not stock_data:
            return f"Could not find data for ticker: {ticker}"
        
        # Get news
        news = self.get_news(ticker, stock_data['company_name'])
        
        # Format the prompt
        prompt = self.format_prompt(stock_data, news)
        
        # Prepare system prompt (empty as requested)
        system_prompt = ""

        # Combine prompts
        input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate prediction
        print("Generating prediction...")
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
            
        # Add search timestamp
        result = {
            "ticker": ticker,
            "current_price": f"${stock_data['current_price']:.2f}",
            "search_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": response,
            "data_sources": ["Yahoo Finance", "NewsAPI" if self.news_client else "Yahoo News"]
        }
        
        return result

# Command-line interface
if __name__ == "__main__":
    import argparse
    import sys
    
    # If no arguments are provided, show example usage
    if len(sys.argv) == 1:
        print("\nStonk Market Predictor - Web Search Interface")
        print("=============================================")
        print("\nUsage examples:")
        print("  python web_search.py AAPL        # Predict Apple stock")
        print("  python web_search.py MSFT        # Predict Microsoft stock")
        print("  python web_search.py GOOGL       # Predict Google stock")
        print("\nSample tickers to try: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM, BAC, DIS")
        print("\nFor more options:")
        print("  python web_search.py --help")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description='Stock Market Prediction with Web Search')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--model', type=str, default='./qwen_stonk_advisor_final',
                        help='Path to fine-tuned model')
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model):
        print(f"\nError: Model path '{args.model}' does not exist.")
        print("If you haven't fine-tuned the model yet, run train_qwen_grpo.py first.")
        
        # Check for partial model
        partial_path = "./qwen_stonk_advisor_partial"
        if os.path.exists(partial_path):
            print(f"\nFound partial model at '{partial_path}'.")
            print(f"You can try using it with: python web_search.py {args.ticker} --model {partial_path}")
        
        # Offer to use base model
        print("\nAlternatively, you can use the base Qwen model directly:")
        print(f"python web_search.py {args.ticker} --model Qwen/Qwen2.5-1.5B-Instruct")
        sys.exit(1)
    
    try:
        engine = StockSearchEngine(model_path=args.model)
        result = engine.predict(args.ticker)
        
        # Pretty print the result
        if isinstance(result, dict):
            print(f"\n===== PREDICTION FOR {result['ticker']} =====")
            print(f"Current Price: {result['current_price']}")
            print(f"Search Time: {result['search_time']}")
            print(f"\nPREDICTION:")
            print(result['prediction'])
            print(f"\nData Sources: {', '.join(result['data_sources'])}")
        else:
            print(result)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Make sure the ticker symbol is valid (e.g., AAPL, MSFT, GOOGL)")
        print("3. Ensure you have enough GPU memory available")
        print("4. Try with a different ticker symbol") 