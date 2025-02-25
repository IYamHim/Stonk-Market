from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from datasets import load_dataset
import json
import numpy as np
from datetime import datetime, timedelta
import re
from predict import load_trained_model, predict_stock
import wandb
import os
from torch.utils.data import DataLoader
import traceback
from transformers import BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling

def format_prompt(row):
    """Format the input data into a prompt"""
    company_info = row['company_info']
    
    context = f"""Stock: {company_info['ticker']}
Date: {company_info['current_date']}
Company: {company_info['company_info']['name']}
Description: {company_info['company_info']['description']}
Current Price: ${company_info['price']['close']:.2f}
Previous Close: ${company_info['price']['close_previous']:.2f}
Price Change: {((company_info['price']['close'] - company_info['price']['close_previous']) / company_info['price']['close_previous'] * 100):.2f}%

Recent News:
{chr(10).join(['- ' + headline for headline in company_info['news']['news_headlines'][:5]])}

Financial Data:
{company_info['financials']['financials']}

Question: Based on this information, analyze whether this stock will go up or down in the next trading day. Provide your reasoning and a specific prediction with a percentage range.
"""
    return context

def get_next_day_price_change(dataset, current_row_idx):
    """Get the actual price change for the next trading day"""
    current_row = dataset[current_row_idx]
    ticker = current_row['company_info']['ticker']
    current_date = datetime.strptime(current_row['company_info']['current_date'], '%Y-%m-%d')
    
    # Look for the next trading day's data
    for i in range(current_row_idx + 1, len(dataset)):
        next_row = dataset[i]
        if next_row['company_info']['ticker'] == ticker:
            next_date = datetime.strptime(next_row['company_info']['current_date'], '%Y-%m-%d')
            if next_date - current_date <= timedelta(days=3):  # Allow for weekends
                next_price = next_row['company_info']['price']['close']
                current_price = current_row['company_info']['price']['close']
                return ((next_price - current_price) / current_price) * 100
    return None

def compute_reward(prediction, actual_change, response_text):
    """Compute reward based on prediction accuracy and response format"""
    reward = 0
    
    # Check format compliance (20% of total reward)
    format_reward = 0
    if re.search(r'<reason>.*?</reason>', response_text) and re.search(r'<answer>.*?</answer>', response_text):
        format_reward = 2.0
    
    # Extract prediction details
    prediction_match = re.search(r'<answer>.*?(up|down).*?(\d+(?:\.\d+)?)\s*%.*?</answer>', response_text.lower())
    if prediction_match:
        direction = prediction_match.group(1)
        predicted_change = float(prediction_match.group(2))
        
        # Direction reward (40% of total reward)
        direction_correct = (direction == 'up' and actual_change > 0) or (direction == 'down' and actual_change < 0)
        direction_reward = 4.0 if direction_correct else -2.0
        
        # Magnitude reward (40% of total reward)
        magnitude_diff = abs(abs(predicted_change) - abs(actual_change))
        magnitude_reward = 4.0 * max(0, 1 - magnitude_diff/5.0)  # Scale down based on difference
        
        reward = format_reward + direction_reward + magnitude_reward
    
    return max(-1, min(10, reward))  # Clip reward between -1 and 10

def process_chunk(examples, idx):
    """Process a chunk of examples"""
    outputs = []
    for i, sample in enumerate(examples):
        try:
            # Print sample structure for debugging
            print(f"Sample structure: {type(sample)}")
            
            system_prompt = """You are an elite stock market analyst and trader with decades of experience in financial markets. You have:
1. A proven track record of accurate stock predictions
2. Deep expertise in technical and fundamental analysis
3. Advanced understanding of market psychology and sentiment analysis
4. Extensive knowledge of global economic factors
5. Experience in quantitative analysis and pattern recognition"""
            
            # Format the user prompt using the existing function
            try:
                user_prompt = format_prompt(sample)
            except Exception as e:
                print(f"Error in format_prompt: {str(e)}")
                print(f"Sample content: {sample}")
                continue
                
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Add expected assistant response for training
            expected_response = """<reason>Based on the provided information, I've analyzed the following key factors:
1. Price momentum and recent performance
2. News sentiment and market impact
3. Financial metrics and company fundamentals
</reason>
<answer>Based on my analysis, I predict the stock will [up/down] by [X-Y]% in the next trading day.</answer>"""
            
            full_prompt += expected_response + "<|im_end|>"
            
            encoded = tokenizer(
                full_prompt,
                truncation=True,
                max_length=2048,
                padding=False,
                return_tensors=None
            )
            
            # Create labels that match input_ids
            labels = encoded["input_ids"].copy()
            
            # Mask the non-assistant part of the input
            assistant_start = full_prompt.find("<|im_start|>assistant\n")
            if assistant_start != -1:
                # Convert the substring before assistant prompt to tokens
                prefix_tokens = len(tokenizer(full_prompt[:assistant_start])["input_ids"])
                # Mask all tokens before assistant response
                labels[:prefix_tokens] = [-100] * prefix_tokens
            
            outputs.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels
            })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} examples in current chunk")
                print(f"Last example length: {len(encoded['input_ids'])}")
            
        except Exception as e:
            print(f"Error processing example {idx + i}: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
            continue
            
    return outputs

def process_single_example(example, tokenizer):
    try:
        # Validate required fields
        if 'ticker' not in example or 'company_info' not in example:
            print(f"Missing required fields. Available keys: {example.keys()}")
            return None
            
        # Use the existing user_prompt if available, otherwise format it
        if 'user_prompt' in example:
            user_prompt = example['user_prompt']
        else:
            try:
                user_prompt = format_prompt(example)
            except Exception as e:
                print(f"Error formatting prompt: {str(e)}")
                return None
        
        system_prompt = """You are an elite stock market analyst and trader with decades of experience in financial markets. You have:
1. A proven track record of accurate stock predictions
2. Deep expertise in technical and fundamental analysis
3. Advanced understanding of market psychology and sentiment analysis
4. Extensive knowledge of global economic factors
5. Experience in quantitative analysis and pattern recognition"""
        
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Create a simple response for training
        expected_response = """<reason>Based on the provided information, I've analyzed the following key factors:
1. Price momentum and recent performance
2. News sentiment and market impact
3. Financial metrics and company fundamentals
</reason>
<answer>Based on my analysis, I predict the stock will up by 1.5% in the next trading day.</answer>"""
        
        full_prompt += expected_response + "<|im_end|>"
        
        # Encode with padding and truncation
        model_inputs = tokenizer(
            full_prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # Create labels matching the input_ids length
        labels = model_inputs["input_ids"].copy()
        
        # Find assistant start token position
        assistant_start = full_prompt.find("<|im_start|>assistant\n")
        if assistant_start != -1:
            # Get tokens before assistant response
            prefix_tokens = tokenizer(
                full_prompt[:assistant_start],
                add_special_tokens=False,
                truncation=True,
                max_length=512
            )["input_ids"]
            # Mask tokens before assistant response
            labels[:len(prefix_tokens)] = [-100] * len(prefix_tokens)
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
        
    except Exception as e:
        print(f"Error processing example: {str(e)}")
        return None

def prepare_training_data(tokenizer, batch_size=1000):
    """Prepare dataset for training in smaller chunks"""
    print("Loading dataset...")
    dataset = load_dataset("2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2", split="train")
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Process all examples
    processed_examples = []
    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(f"\nProcessing example {i}/{len(dataset)}")
        
        processed = process_single_example(example, tokenizer)
        if processed is not None:
            processed_examples.append(processed)
            
            if len(processed_examples) == 1:
                print("\nFirst example processed successfully!")
                print(f"Input IDs length: {len(processed['input_ids'])}")
                print(f"Labels length: {len(processed['labels'])}")
        
        if len(processed_examples) >= 10000:  # Limit for testing
            print("Reached example limit")
            break
    
    print(f"\nSuccessfully processed {len(processed_examples)} examples")
    
    if len(processed_examples) == 0:
        raise ValueError("No examples were successfully processed! Check the error messages above.")
    
    from datasets import Dataset
    return Dataset.from_list(processed_examples)

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
        reserved = torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024)  # Convert to GB
        print(f"GPU Memory allocated: {allocated:.2f}GB")
        print(f"GPU Memory reserved: {reserved:.2f}GB")
        
        # Print per-device memory
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} Memory:")
            print(f"  Total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f}GB")
            print(f"  Free: {torch.cuda.memory_reserved(i) / (1024**3):.2f}GB")
            print(f"  Used: {torch.cuda.memory_allocated(i) / (1024**3):.2f}GB")

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")

def split_long_sequence(text, tokenizer, max_length=2048):
    # First tokenize without truncation
    tokens = tokenizer(text, truncation=False)
    
    # Split into chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens.input_ids:
        if current_length + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = [token]
            current_length = 1
        else:
            current_chunk.append(token)
            current_length += 1
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def train_model():
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=512,
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create custom data collator
    class CustomDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def __call__(self, examples):
            # Ensure all sequences are the same length
            max_length = max(len(ex["input_ids"]) for ex in examples)
            
            # Pad all sequences to max_length
            input_ids = [ex["input_ids"] + [self.tokenizer.pad_token_id] * (max_length - len(ex["input_ids"])) for ex in examples]
            attention_mask = [[1] * len(ex["input_ids"]) + [0] * (max_length - len(ex["input_ids"])) for ex in examples]
            labels = [ex["labels"] + [-100] * (max_length - len(ex["labels"])) for ex in examples]
            
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)
            }
    
    # PHASE 1: Supervised fine-tuning parameters
    sft_training_args = TrainingArguments(
        output_dir="./results/sft",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-4,
        warmup_ratio=0.01,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        max_grad_norm=1.0,
        logging_first_step=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
        group_by_length=True,
    )
    
    # Initialize model with optimized settings
    print("Initializing model...")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA with optimized settings
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Process and split data with smaller validation set
    print("Preparing training data...")
    processed_dataset = prepare_training_data(tokenizer)
    train_test_split = processed_dataset.train_test_split(test_size=0.05)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # PHASE 1: Standard supervised fine-tuning
    print("Starting supervised fine-tuning (SFT)...")
    sft_trainer = Trainer(
        model=model,
        args=sft_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CustomDataCollator(tokenizer),
    )
    
    try:
        # Train the model using SFT
        sft_trainer.train()
        print("Supervised fine-tuning completed successfully")
        
        # Save the SFT model
        sft_save_path = "./qwen_stock_advisor_sft"
        model.save_pretrained(sft_save_path)
        tokenizer.save_pretrained(sft_save_path)
        print(f"SFT model saved to: {os.path.abspath(sft_save_path)}")
        
        # PHASE 2: GRPO training
        print("\nStarting GRPO training (reinforcement learning with rewards)...")
        
        # Create a reference model (frozen copy of SFT model)
        from copy import deepcopy
        ref_model = deepcopy(model)
        
        # Freeze reference model weights
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        
        # Load original stock dataset for reward computation
        print("Loading stock data for GRPO training...")
        stock_dataset = load_dataset("2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2", split="train")
        
        # GRPO hyperparameters
        grpo_epochs = 2
        batch_size = 8
        grpo_lr = 1e-5
        kl_coef = 0.1
        clip_range = 0.2
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=grpo_lr)
        
        for epoch in range(grpo_epochs):
            print(f"\nGRPO Epoch {epoch+1}/{grpo_epochs}")
            model.train()
            
            # Sample examples for this epoch (limit for efficiency)
            epoch_samples = min(200, len(stock_dataset))
            indices = np.random.choice(len(stock_dataset), epoch_samples, replace=False)
            
            total_reward = 0
            total_samples = 0
            
            # Process in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_rewards = []
                batch_losses = []
                
                # Collect experiences for this batch
                for idx in batch_indices:
                    try:
                        # Get the sample and format prompt
                        sample = stock_dataset[idx]
                        user_prompt = format_prompt(sample)
                        
                        system_prompt = """You are an elite stock market analyst and trader with decades of experience in financial markets."""
                        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                        
                        # Get actual price change (ground truth)
                        actual_change = get_next_day_price_change(stock_dataset, idx)
                        if actual_change is None:
                            continue
                        
                        # Tokenize prompt
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        
                        # Generate response from current model
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=200,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.95,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        
                        # Get generated text
                        response_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                        
                        # Extract prediction from response
                        prediction_match = re.search(r'<answer>.*?(up|down).*?(\d+(?:\.\d+)?)\s*%.*?</answer>', response_text.lower())
                        if prediction_match:
                            direction = prediction_match.group(1)
                            predicted_change = float(prediction_match.group(2))
                            prediction = {"direction": direction, "magnitude": predicted_change}
                        else:
                            prediction = None
                        
                        # Compute reward using existing function
                        reward = compute_reward(prediction, actual_change, response_text)
                        batch_rewards.append(reward)
                        
                        # Forward pass with input + target for RL update
                        target_ids = tokenizer(response_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                        
                        # Get current model log probabilities
                        outputs = model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                        )
                        logits = outputs.logits[:, -1, :]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        
                        # Get reference model log probabilities
                        with torch.no_grad():
                            ref_outputs = ref_model(
                                input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                            )
                            ref_logits = ref_outputs.logits[:, -1, :]
                            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                        
                        # Compute KL divergence
                        kl_div = torch.nn.functional.kl_div(
                            log_probs,
                            ref_log_probs,
                            reduction='batchmean'
                        )
                        
                        # Policy gradient loss with reward
                        pg_loss = -reward * log_probs[0, target_ids[0, 0]]
                        
                        # Add KL penalty
                        loss = pg_loss + kl_coef * kl_div
                        batch_losses.append(loss)
                        
                        total_samples += 1
                        total_reward += reward
                        
                    except Exception as e:
                        print(f"Error processing sample {idx}: {str(e)}")
                        continue
                
                # Update model if we have valid losses
                if len(batch_losses) > 0:
                    optimizer.zero_grad()
                    # Sum losses and backward
                    batch_loss = torch.stack(batch_losses).mean()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
                    print(f"Batch {batch_start//batch_size + 1} - Average Reward: {avg_reward:.2f}")
            
            # Epoch summary
            if total_samples > 0:
                avg_epoch_reward = total_reward / total_samples
                print(f"Epoch {epoch+1} summary - Average Reward: {avg_epoch_reward:.2f}, Samples: {total_samples}")
                
                # Save checkpoint after each epoch
                grpo_save_path = f"./qwen_stock_advisor_grpo_epoch_{epoch+1}"
                model.save_pretrained(grpo_save_path)
                print(f"GRPO model (epoch {epoch+1}) saved to: {os.path.abspath(grpo_save_path)}")
        
        # Save final model after GRPO
        final_save_path = "./qwen_stock_advisor_final"
        model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        print(f"Final GRPO model saved to: {os.path.abspath(final_save_path)}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        
        print("Attempting to save partial model...")
        try:
            partial_path = "./qwen_stock_advisor_partial"
            model.save_pretrained(partial_path)
            tokenizer.save_pretrained(partial_path)
            print(f"Partial model saved to: {os.path.abspath(partial_path)}")
        except Exception as save_error:
            print(f"Failed to save partial model: {str(save_error)}")
    
    return model, tokenizer

def test_model(model, tokenizer):
    print("\nTesting model...")
    
    system_prompt = """You are an elite stock market analyst and trader with decades of experience in financial markets. You have:
1. A proven track record of accurate stock predictions
2. Deep expertise in technical and fundamental analysis
3. Advanced understanding of market psychology and sentiment analysis
4. Extensive knowledge of global economic factors
5. Experience in quantitative analysis and pattern recognition

Your analysis should be thorough and precise, considering multiple factors:
- Technical indicators and price patterns
- Company fundamentals and financial metrics
- Market sentiment and news impact
- Industry trends and competitive position
- Macroeconomic factors

Always structure your response with:
1. <reason> tag containing detailed, multi-factor analysis
2. <answer> tag with a specific directional prediction and percentage range

Your predictions should be well-reasoned and based on concrete data points from the provided information."""

    test_prompt = """Stock: AAPL
Date: 2024-03-14
Company: Apple Inc.
Current Price: $170.50
Previous Close: $168.82

Recent News:
- Apple announces new AI features for iPhone
- Strong iPhone sales in Asian markets
- Partnership with major AI research lab announced

Question: Based on this information, analyze whether this stock will go up or down in the next trading day. Provide your reasoning and a specific prediction with a percentage range."""
    
    input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(
        input_text, 
        return_tensors="pt",
        max_length=8192,  # Increase max length
        truncation=True,
        padding="max_length"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=8192,  # Increase output length
        temperature=0.5,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nModel Response:")
    print(response)

if __name__ == "__main__":
    print("Initial GPU memory:")
    print_gpu_memory()
    
    model, tokenizer = train_model()
    
    print("\nFinal GPU memory:")
    print_gpu_memory()
    
    test_model(model, tokenizer)

    # Get prediction for any stock
    prediction = predict_stock(
        model, 
        tokenizer,
        ticker="MSFT",
        current_price=425.22,
        previous_close=421.45,
        news_headlines=[
            "Microsoft Cloud revenue grows 30%",
            "New AI features announced for Office",
            "Strong enterprise adoption of Azure"
        ]
    )
    print(prediction) 
