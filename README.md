# Disclaimer:

# The information provided here is for testing and educational purposes only and should not be construed as financial advice. Please consult with a licensed financial advisor before making any financial decisions.

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
python Stonk_Trader.py
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
Note: If you're not using a GPU, you may need to modify the environment.yml file to remove CUDA dependencies.

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

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (an example is provided in the Appendix below).

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and Derivative Works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control systems,
   and issue tracking systems that are managed by, or on behalf of, the
   Licensor for the purpose of discussing and improving the Work, but
   excluding communication that is conspicuously marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution."

   "Contributor" shall mean Licensor and any individual or Legal Entity
   on behalf of whom a Contribution has been received by Licensor and
   subsequently incorporated within the Work.

2. Grant of Copyright License. 
   Subject to the terms and conditions of this License, each Contributor
   hereby grants to You a perpetual, worldwide, non-exclusive, no-charge,
   royalty-free, irrevocable copyright license to reproduce, prepare Derivative
   Works of, publicly display, publicly perform, sublicense, and distribute
   the Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. 
   Subject to the terms and conditions of this License, each Contributor
   hereby grants to You a perpetual, worldwide, non-exclusive, no-charge,
   royalty-free, irrevocable (except as stated in this section) patent license
   to make, have made, use, offer to sell, sell, import, and otherwise transfer
   the Work, where such license applies only to those patent claims
   licensable by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s) with the
   Work to which such Contribution(s) was submitted. If You institute patent
   litigation against any entity (including a cross-claim or counterclaim in a
   lawsuit) alleging that the Work or a Contribution incorporated within the
   Work constitutes direct or contributory patent infringement, then any patent
   licenses granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. 
   You may reproduce and distribute copies of the Work or Derivative Works thereof
   in any medium, with or without modifications, and in Source or Object form,
   provided that You meet the following conditions:
   
   (a) You must give any other recipients of the Work or Derivative Works a copy of
   this License; and
   
   (b) You must cause any modified files to carry prominent notices stating that
   You changed the files; and
   
   (c) You must retain, in the Source form of any Derivative Works that You distribute,
   all copyright, patent, trademark, and attribution notices from the Source form of
   the Work, excluding those notices that do not pertain to any part of the Derivative
   Works; and
   
   (d) If the Work includes a "NOTICE" text file as part of its distribution,
