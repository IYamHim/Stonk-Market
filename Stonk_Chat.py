import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import traceback
import sys
import datetime

# Add the parent directory to the path so we can import web_search module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from web_search import StockSearchEngine
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    print("Warning: web_search.py module not found. Web search functionality will be disabled.")
    WEB_SEARCH_AVAILABLE = False

# ===== CONFIGURATION =====
# Model paths and settings
MODEL_PATH = "[path to saved model]"  # Path to saved model
PARTIAL_MODEL_PATH = "[path to partial model]"  # Path to partial model
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Original base model
IS_LORA = True  # Set to True if you fine-tuned with LoRA

# Generation settings
MAX_NEW_TOKENS = 512  # Max length of generated response
TEMPERATURE = 0.7  # Higher = more creative, lower = more deterministic
TOP_P = 0.9  # Nucleus sampling parameter

# Interface settings
SHARE_PUBLIC = True  # Create public link with Gradio
# ========================

def load_model():
    print("Loading model and tokenizer...")
    
    # Check for available models
    model_exists = os.path.exists(MODEL_PATH)
    partial_model_exists = os.path.exists(PARTIAL_MODEL_PATH)
    
    if not model_exists and not partial_model_exists:
        print(f"Warning: Neither {MODEL_PATH} nor {PARTIAL_MODEL_PATH} exists.")
        print(f"Falling back to base model: {BASE_MODEL_NAME}")
        
        # Load base model only
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
                padding_side='left',
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            print("Successfully loaded base model without fine-tuning.")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading base model: {str(e)}")
            traceback.print_exc()
            raise
    
    # Try to load fine-tuned model
    try:
        # Determine which model path to use
        actual_model_path = MODEL_PATH if model_exists else PARTIAL_MODEL_PATH
        print(f"Loading model from: {actual_model_path}")
        
        if IS_LORA:
            print("Loading as LoRA adapter...")
            # Load tokenizer from base model
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
                padding_side='left',
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Check if adapter_config.json exists in the model path
            if os.path.exists(os.path.join(actual_model_path, "adapter_config.json")):
                # Load adapter
                model = PeftModel.from_pretrained(base_model, actual_model_path)
                print("Successfully loaded LoRA adapter.")
            else:
                print(f"Warning: adapter_config.json not found in {actual_model_path}")
                print("Using base model without fine-tuning.")
                model = base_model
        else:
            print("Loading as full model...")
            tokenizer = AutoTokenizer.from_pretrained(
                actual_model_path,
                trust_remote_code=True,
                padding_side='left',
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        model.eval()  # Set model to evaluation mode
        return model, tokenizer
    except Exception as e:
        print(f"Error loading fine-tuned model: {str(e)}")
        print("Falling back to base model...")
        
        try:
            # Load base model as fallback
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
                padding_side='left',
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            print("Successfully loaded base model as fallback.")
            return model, tokenizer
        except Exception as fallback_error:
            print(f"Error loading fallback model: {str(fallback_error)}")
            traceback.print_exc()
            raise

def generate_response(model, tokenizer, message, chat_history):
    # Empty system prompt (removed as requested)
    system_prompt = ""

    # Format the prompt with chat history
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    for user_msg, assistant_msg in chat_history:
        full_prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        full_prompt += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    
    full_prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_prefix = "<|im_start|>assistant\n"
        assistant_suffix = "<|im_end|>"
        
        last_assistant_start = full_response.rfind(assistant_prefix)
        if last_assistant_start != -1:
            response_start = last_assistant_start + len(assistant_prefix)
            response_end = full_response.find(assistant_suffix, response_start)
            
            if response_end != -1:
                response = full_response[response_start:response_end].strip()
            else:
                response = full_response[response_start:].strip()
        else:
            response = full_response.strip()
        
        return response
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I encountered an error processing your request. Please try again."

# New function for live stock search
def search_stock_data(ticker, model, tokenizer):
    if not WEB_SEARCH_AVAILABLE:
        return f"Web search functionality is not available. Please install the required dependencies and ensure web_search.py is accessible."
    
    try:
        # Check if we're using a base model or fine-tuned model
        using_base_model = not (os.path.exists(MODEL_PATH) or os.path.exists(PARTIAL_MODEL_PATH))
        
        if using_base_model:
            # If using base model, don't try to load as PEFT
            from web_search import StockSearchEngine
            
            # Create a modified version of StockSearchEngine that doesn't use PEFT
            class DirectStockSearchEngine(StockSearchEngine):
                def __init__(self, model_path):
                    print("Initializing Direct Stock Search Engine...")
                    # Skip the parent class __init__ and implement our own
                    self.tokenizer = tokenizer  # Use the already loaded tokenizer
                    self.model = model  # Use the already loaded model
                    
                    # Initialize news API client (copied from parent)
                    self.news_client = None
                    try:
                        from newsapi import NewsApiClient
                        NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"
                        if NEWSAPI_KEY != "YOUR_NEWSAPI_KEY":
                            self.news_client = NewsApiClient(api_key=NEWSAPI_KEY)
                            print("NewsAPI client initialized successfully.")
                    except:
                        print("Warning: NewsAPI not available. Using fallback news sources.")
                    
                    print("Direct Stock Search Engine initialized successfully!")
            
            # Use our custom engine with the already loaded model
            search_engine = DirectStockSearchEngine(BASE_MODEL_NAME)
        else:
            # Use the normal StockSearchEngine with the fine-tuned model path
            model_to_use = MODEL_PATH if os.path.exists(MODEL_PATH) else PARTIAL_MODEL_PATH
            
            # Get the absolute path to make sure it's found
            abs_model_path = os.path.abspath(model_to_use)
            print(f"Using model at absolute path: {abs_model_path}")
            
            search_engine = StockSearchEngine(model_path=abs_model_path)
        
        # Get prediction
        result = search_engine.predict(ticker)
        
        # Format the output for display
        if isinstance(result, dict):
            output = f"## 📊 LIVE DATA FOR {result['ticker']} 📊\n\n"
            output += f"**Current Price:** {result['current_price']}\n"
            output += f"**Search Time:** {result['search_time']}\n\n"
            output += f"### PREDICTION:\n{result['prediction']}\n\n"
            output += f"*Data Sources: {', '.join(result['data_sources'])}*"
            return output
        else:
            return f"Error searching for {ticker}: {result}"
    except Exception as e:
        traceback.print_exc()
        return f"Error searching for {ticker}: {str(e)}"

def main():
    print("Initializing Stock Advisor Chat Interface...")
    
    # Load model
    model, tokenizer = load_model()
    
    # Define chat function for Gradio
    def chat(message, history):
        response = generate_response(model, tokenizer, message, history)
        return response
    
    # Define stock search function
    def search_stock(ticker):
        if not ticker or ticker.strip() == "":
            return "Please enter a valid ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        
        ticker = ticker.strip().upper()
        return search_stock_data(ticker, model, tokenizer)
    
    # Create Gradio interface
    with gr.Blocks(title="AI Stonk Advisor") as demo:
        gr.Markdown("# 🚀 AI Stonk Advisor with Live Data")
        gr.Markdown("Ask questions about stonks, market trends, or use the search tool to get live data and predictions.")
        
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.ChatInterface(
                    fn=chat,
                    examples=[
                        "What do you think about AAPL stonk? Current price is $175.50, previous close was $173.25. Recent news includes new iPhone release and services growth.",
                        "Analyze MSFT stonk. Current price: $410.20, previous: $405.75. News: Cloud revenue up 30%, new AI features, strong enterprise adoption.",
                        "Is NVDA a good buy right now? Price: $950, previous: $930. News: New GPU launch, AI demand increasing, data center growth accelerating."
                    ],
                )
            
            with gr.Tab("Live Stonk Search"):
                gr.Markdown("## 🔍 Search for Real-Time Stonk Data and Predictions")
                
                ticker_input = gr.Textbox(
                    label="Enter Stonk Ticker Symbol",
                    placeholder="e.g., AAPL, MSFT, GOOGL",
                    info="Enter a stonk ticker symbol to get live data and AI prediction"
                )
                
                search_button = gr.Button("Search", variant="primary")
                
                with gr.Accordion("Popular Tickers", open=False):
                    gr.Markdown("""
                    - **Tech**: AAPL (Apple), MSFT (Microsoft), GOOGL (Alphabet), META (Meta), AMZN (Amazon)
                    - **EV/Auto**: TSLA (Tesla), F (Ford), GM (General Motors)
                    - **Finance**: JPM (JP Morgan), BAC (Bank of America), V (Visa)
                    - **Retail**: WMT (Walmart), TGT (Target), COST (Costco)
                    - **Entertainment**: DIS (Disney), NFLX (Netflix)
                    """)
                
                result_md = gr.Markdown("Results will appear here after search")
                
                search_button.click(
                    fn=search_stock,
                    inputs=ticker_input,
                    outputs=result_md,
                )
        
        gr.Markdown("### 📝 About")
        gr.Markdown(
            """This AI assistant uses the Qwen2.5-1.5B model fine-tuned on stonk market data. 
            The Live Stonk Search feature retrieves real-time data from Yahoo Finance and generates predictions 
            based on current market conditions. All predictions are for educational purposes only and should not 
            be considered financial advice."""
        )
    
    # Launch the interface
    demo.launch(share=SHARE_PUBLIC)
    print("Interface closed.")

if __name__ == "__main__":
    main()
