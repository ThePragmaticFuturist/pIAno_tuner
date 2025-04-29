import argparse
import os
import logging
import torch
import yaml # For parsing target_modules list easily
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging as hf_logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
hf_logging.set_verbosity_info() # Set transformers logging level

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a local LLM using QLoRA")

    # Model Arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Identifier of the pre-trained model from Hugging Face Hub or path to local model directory.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action='store_true',
        help="Set this flag if the model requires executing custom code (e.g., Phi models)."
    )

    # Data Arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the training data file (JSON Lines format, e.g., training_data.jsonl).",
    )
    parser.add_argument(
        "--dataset_text_field",
        type=str,
        default="text",
        help="The name of the column in the dataset containing the formatted text.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024, # Adjust based on model/VRAM
        help="Maximum sequence length for tokenization.",
    )

    # QLoRA Arguments
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha scaling parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]", # Common for Llama/Mistral style
        help="Comma-separated list or YAML list string of module names to apply LoRA to (e.g., '[q_proj,v_proj]'). Check model architecture.",
    )
    parser.add_argument(
        "--use_4bit",
        action='store_true',
        default=True, # Enabled by default for QLoRA
        help="Load model in 4-bit precision.",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bfloat16", # Use bfloat16 for Ampere+ GPUs
        help="Compute dtype for 4-bit base models (bfloat16 or float16).",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4", # Recommended quant type
        help="Quantization type (fp4 or nf4).",
    )
    parser.add_argument(
        "--use_nested_quant",
        action='store_true',
        default=True, # Enable nested quantization
        help="Activate nested quantization for 4-bit base models.",
    )


    # Training Arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the fine-tuned adapters and training artifacts.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer to use (e.g., paged_adamw_8bit, adamw_hf).")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save adapter checkpoints every N steps.")
    parser.add_argument("--fp16", action='store_true', help="Enable FP16 mixed precision (alternative to BF16).")
    parser.add_argument("--bf16", action='store_true', default=True, help="Enable BF16 mixed precision (requires Ampere+ GPU).") # Defaulting to True if available
    parser.add_argument("--gradient_checkpointing", action='store_true', default=True, help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--packing", action='store_true', help="Pack multiple short sequences into one for efficiency.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to (tensorboard, wandb, etc.).")

    args = parser.parse_args()

    # Post-process lora_target_modules if provided as a string list
    try:
        # Safely evaluate if it looks like a list (e.g., "[a, b]") or use YAML load
        if args.lora_target_modules.strip().startswith('[') and args.lora_target_modules.strip().endswith(']'):
             # Attempt to parse as a list-like string
            args.lora_target_modules = yaml.safe_load(args.lora_target_modules)
        elif ',' in args.lora_target_modules:
             # Assume comma-separated if not list-like
            args.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
        else:
             # Assume single module if no comma and not list-like
            args.lora_target_modules = [args.lora_target_modules.strip()]

        if not isinstance(args.lora_target_modules, list):
             raise ValueError("Could not parse lora_target_modules into a list.")
        logging.info(f"Parsed LoRA target modules: {args.lora_target_modules}")
    except Exception as e:
        logging.error(f"Error parsing --lora_target_modules '{args.lora_target_modules}': {e}")
        logging.error("Please provide it as a comma-separated string (e.g., q_proj,v_proj) or a YAML list string (e.g., '[q_proj, v_proj]').")
        exit(1)


    # Validate compute dtype
    if args.bnb_4bit_compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        logging.warning("BF16 is not supported on this GPU. Switching compute dtype to FP16.")
        args.bnb_4bit_compute_dtype = "float16"
        args.bf16 = False
        args.fp16 = True # Enable fp16 if bf16 is not available and user didn't explicitly set fp16=False

    if args.bf16 and args.fp16:
        logging.warning("Both --bf16 and --fp16 are set. Prioritizing BF16 if supported, otherwise using FP16.")
        if not torch.cuda.is_bf16_supported():
             args.bf16 = False


    return args

def main():
    args = parse_arguments()

    if not torch.cuda.is_available():
        logging.error("CUDA is not available. QLoRA requires a CUDA-enabled GPU. Exiting.")
        return

    logging.info(f"Starting fine-tuning process with arguments: {args}")

    # --- 1. Load Dataset ---
    logging.info(f"Loading dataset from {args.dataset_path}")
    try:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        logging.info(f"Dataset loaded successfully with {len(dataset)} examples.")
        # Optional: Add split for evaluation
        # if "validation_split_percentage" in args:
        #     dataset = dataset.train_test_split(test_size=args.validation_split_percentage / 100.0)
        #     train_dataset = dataset["train"]
        #     eval_dataset = dataset["test"]
        # else:
        train_dataset = dataset
        eval_dataset = None # Set to None if no eval split
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # --- 2. Configure Quantization (QLoRA) ---
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )
    logging.info(f"BitsAndBytesConfig configured: {bnb_config}")

    # --- 3. Load Base Model ---
    logging.info(f"Loading base model: {args.model_name_or_path}")
    # Check potential memory footprint issue with device_map
    if torch.cuda.device_count() > 1:
        device_map = "auto" # Let accelerate handle multi-GPU distribution
        logging.info(f"Multiple GPUs detected ({torch.cuda.device_count()}). Using device_map='auto'.")
    else:
        # device_map = {"": 0} # Explicitly map to GPU 0 if single GPU
        device_map = "auto" # Let accelerate handle single GPU too
        logging.info("Single GPU detected. Using device_map='auto'.")


    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map=device_map, # Automatically map layers to devices (GPU/CPU/RAM)
            trust_remote_code=args.trust_remote_code,
        )
        model.config.use_cache = False # Crucial for training stability
        # Mitigate gradient checkpointing issue / potential warning
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # Alternative if the above method isn't available or needed for older PEFT/Transformers
             def make_inputs_require_grad(module, input, output):
                  output.requires_grad_(True)
             if args.gradient_checkpointing:
                   model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


        # Needed for some models like Phi that might have tensor parallelism conflicts
        # model.config.pretraining_tp = 1 # Set to 1 to avoid issues during training - uncomment if needed

        logging.info("Base model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load base model: {e}")
        logging.exception("Detailed traceback:") # Print full traceback
        return

    # --- 4. Load Tokenizer ---
    logging.info(f"Loading tokenizer for: {args.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        # Set padding token if unset
        if tokenizer.pad_token is None:
            logging.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Must be right for training causal LMs with SFTTrainer
        logging.info(f"Tokenizer loaded. Pad token: {tokenizer.pad_token}, Padding side: {tokenizer.padding_side}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return

    # --- 5. Configure LoRA ---
    logging.info("Preparing model for K-bit training and configuring LoRA...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logging.info(f"LoRA Config: {peft_config}")

    try:
        model = get_peft_model(model, peft_config)
        logging.info("LoRA adapters applied to the model.")
        model.print_trainable_parameters() # Show % of trainable parameters
    except ValueError as e:
         logging.error(f"Error applying LoRA. Check if --lora_target_modules are correct for {args.model_name_or_path}. Error: {e}")
         logging.info("You might need to inspect the model architecture (e.g., print(model)) to find the correct layer names.")
         return
    except Exception as e:
        logging.error(f"An unexpected error occurred while applying LoRA: {e}")
        return

    # --- 6. Configure Training Arguments ---
    logging.info("Configuring Training Arguments...")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_strategy="steps", # Save checkpoints at regular intervals
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16 if not args.bf16 else False, # Only one can be true
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=0.3, # Common value for QLoRA
        max_steps=-1, # -1 means use num_train_epochs
        group_by_length=True, # Speeds up training by grouping similar length sequences
        report_to=args.report_to,
        # evaluation_strategy="steps" if eval_dataset else "no", # Enable eval if eval_dataset exists
        # eval_steps=args.save_steps if eval_dataset else None, # Evaluate at the same frequency as saving
        save_total_limit=2, # Keep only the last 2 checkpoints
        load_best_model_at_end=False, # Can be True if evaluation_strategy is enabled
        # dataloader_num_workers=4 # Can adjust based on CPU cores
    )
    logging.info(f"Training Arguments: {training_arguments}")

    # --- 7. Initialize Trainer ---
    logging.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Pass evaluation dataset if available
        peft_config=peft_config,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing, # Use packing if specified
    )
    logging.info("SFTTrainer initialized.")

    # --- 8. Start Training ---
    logging.info("*** Starting Training ***")
    try:
        train_result = trainer.train()
        logging.info("Training finished successfully.")

        # --- 9. Save Final Model & Metrics ---
        final_adapter_dir = os.path.join(args.output_dir, "final_adapters")
        logging.info(f"Saving final LoRA adapters to: {final_adapter_dir}")
        trainer.save_model(final_adapter_dir) # Saves only the LoRA adapters

        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logging.info(f"Training metrics saved.")

        # Optionally save the tokenizer as well
        tokenizer.save_pretrained(final_adapter_dir)
        logging.info(f"Tokenizer saved to: {final_adapter_dir}")

        logging.info("--- Fine-tuning Complete ---")

    except Exception as e:
        logging.error("An error occurred during training.")
        logging.exception("Detailed traceback:") # Print full traceback

if __name__ == "__main__":
    main()
