# Stonk-Market

GRPO Trainer for Stonks powered by Qwen.

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

This repository contains code for training a Qwen2.5-1.5B model to predict stonk market movements using LoRA fine-tuning.

## Overview

The script `train_qwen_grpo.py` fine-tunes the Qwen2.5-1.5B language model to analyze stonk market data and make next-day price movement predictions. The model is trained to:

1. Analyze company information, price data, news, and financials
2. Provide reasoned analysis in a structured format
3. Make specific price movement predictions with percentage ranges

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

1. **Data Processing**
   - Formats input data into structured prompts
   - Handles tokenization and padding
   - Creates masked labels for assistant responses

2. **Model Configuration**
   - Uses 4-bit quantization for memory efficiency
   - Implements LoRA for efficient fine-tuning
   - Configures gradient checkpointing and mixed precision

3. **Training Parameters**
   - Batch size: 2
   - Gradient accumulation steps: 16
   - Learning rate: 2e-4
   - Max sequence length: 512
   - Uses left-side padding
   - Custom data collation

## Usage

1. Train the model:

```bash
python train_qwen_grpo.py
```

2. The script will:
   - Load and process the dataset
   - Initialize the model with optimized settings
   - Train using the specified parameters
   - Save checkpoints during training
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
- Final model: `./qwen_stonk_advisor_final/`
- Partial model (if interrupted): `./qwen_stonk_advisor_partial/`
- Training checkpoints: `./results/`

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
