# Disclaimer:

# The information provided here is for testing and educational purposes only and should not be construed as financial advice. Please consult with a licensed financial advisor before making any financial decisions.

# Stonk Trader

A trading model powered by Qwen with GRPO (Generative Reinforcement from Preference Optimization).

## Environment Setup

### Option 1: Using Conda (Recommended)

1. Install [Conda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already

2. Create the conda environment from the environment.yml file:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate stonk_trading
```

4. Verify the installation (Optional):

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### Option 2: Using Python Virtual Environment

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Verify the installation:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

Note: For GPU support with pip installation, you might need to install PyTorch with CUDA separately following the instructions from [PyTorch's website](https://pytorch.org/get-started/locally/).

# Stonk Market Prediction Model Training

This repository contains code for training a Qwen2.5-1.5B model to predict stonk market movements using a two-phase approach: supervised fine-tuning followed by GRPO reinforcement learning.

## Overview

The script `train/grpo.py` implements a two-phase training approach:

1. **Supervised Fine-Tuning (SFT)**: Initial training on stock prediction examples
2. **GRPO Training**: Reinforcement learning optimization using a reward function that evaluates:
   - Format compliance (20% of reward)
   - Direction prediction accuracy (40% of reward) 
   - Magnitude prediction accuracy (40% of reward)

The model is trained to:
1. Analyze company information, price data, news, and financials
2. Provide reasoned analysis in a structured format
3. Make specific price movement predictions with percentage ranges
4. Optimize for prediction accuracy using reinforcement learning

## Requirements

Required packages:
- transformers>=4.36.0
- torch>=2.0.0
- peft>=0.7.0
- datasets>=2.14.0
- bitsandbytes>=0.41.0
- accelerate>=0.25.0

## Hardware Requirements

- GPU with at least 8GB VRAM
- 16GB+ system RAM recommended
- CUDA compatible GPU

## Dataset

The model uses the "2084collective/deepstock-sp500-companies-with-info-and-user-prompt" dataset which includes:
- Company information
- Stonk prices
- News headlines
- Financial metrics
- Pre-formatted prompts

## Training Process

1. **Phase 1: Supervised Fine-Tuning**
   - Formats input data into structured prompts
   - Handles tokenization and padding
   - Creates masked labels for assistant responses
   - Uses standard cross-entropy loss

2. **Phase 2: GRPO Training**
   - Creates a reference model (frozen copy of SFT model)
   - Generates predictions and evaluates them with the reward function
   - Uses policy gradient updates with KL divergence penalty
   - Optimizes for prediction accuracy while preventing model drift

3. **Model Configuration**
   - Uses 4-bit quantization for memory efficiency
   - Implements LoRA for efficient fine-tuning
   - Configures gradient checkpointing and mixed precision

4. **Training Parameters**
   - SFT: Batch size 4, gradient accumulation steps 8, learning rate 5e-4
   - GRPO: Batch size 1, learning rate 1e-5, KL coefficient 0.1
   - Max sequence length: 512
   - Uses left-side padding
   - Custom data collation

## Usage

1. Train the model:

```bash
python train/grpo.py
```

2. The script will:
   - Load and process the dataset
   - Perform supervised fine-tuning
   - Save the SFT checkpoint
   - Perform GRPO training with the reward function
   - Save checkpoints after each epoch
   - Save the final model or partial model if interrupted

## Output Format

The model generates responses in a structured format:
```
<reason>
Detailed analysis of factors including:
1. Price momentum
2. News sentiment
3. Financial metrics
</reason>
<answer>
Specific prediction with direction and percentage range
</answer>
```

## Model Artifacts

The script saves:
- SFT model: `./qwen_stonk_advisor_sft/`
- GRPO epoch checkpoints: `./qwen_stonk_advisor_grpo_epoch_N/`
- Final model: `./qwen_stonk_advisor_final/`
- Partial model (if interrupted): `./qwen_stonk_advisor_partial/`

## Testing

The script includes a test function that:
1. Loads the trained model
2. Runs inference on a sample prompt
3. Prints the model's analysis and prediction

## Error Handling

- Validates input data format
- Handles training interruptions gracefully
- Saves partial progress when possible
- Provides detailed error messages

## Memory Management

- Uses 4-bit quantization
- Implements gradient checkpointing
- Optimizes sequence length and batch size
- Monitors GPU memory usage

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your License Here]

Note: If you're not using a GPU, you may need to modify the environment.yml file to remove CUDA dependencies.

## Web Search Integration

The model includes a web search capability for fetching real-time stock data and making predictions.

### Installation

Install additional dependencies:
```bash
pip install yfinance requests beautifulsoup4 newsapi-python
```

For news retrieval, you'll need a NewsAPI key (optional):
1. Get a free API key from [NewsAPI](https://newsapi.org/)
2. Update the `NEWSAPI_KEY` variable in `web_search.py`

### Usage

Search for and predict a stock using the command line:
```bash
python web_search.py AAPL
```

Or import into your own code:
```python
from web_search import StockSearchEngine

engine = StockSearchEngine()
result = engine.predict("AAPL")
print(result["prediction"])
```

### Features

- Real-time stock data from Yahoo Finance
- Recent news articles from NewsAPI or Yahoo Finance
- Financial metrics and company information
- Structured predictions with reasoning

## License

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
