import os
import sys
import time
import logging
import json
import argparse
from pathlib import Path

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

# Resolve key paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ROOT_PARENT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "code"

# Ensure both the package root and code dir are importable
for path in (ROOT_PARENT, CODE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from storys.code.utils.prompts import get_prompt_template

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = BASE_DIR.parent
TRANSFORMER_ROOT = PROJECT_ROOT.parent / "transformer-models"
DEFAULT_DATA_FILE = (PROJECT_ROOT / "data" / "stories_dev.jsonl").resolve()
TEMP_OUTPUT_DIR = (BASE_DIR / "outputs" / "temp_output").resolve()
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_generation_model(base_model, checkpoint_root: str, use_base_model_only: bool = False) -> tuple:
    """Load the model for generating feedback."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        peft_type="LORA",
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = GRPOConfig(
        output_dir=str(TEMP_OUTPUT_DIR),
        per_device_train_batch_size=1,
        log_completions=True,
        max_completion_length=1024,
        max_prompt_length=2048,
        scale_rewards=False,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        bf16=True,
        label_names=[],
        use_vllm=True,
        vllm_mode="colocate",
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        use_cpu=not torch.cuda.is_available(),
    )

    empty_reward = lambda *args, **kwargs: 0.0

    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=empty_reward,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    if not use_base_model_only:
        trainer.model.load_adapter(checkpoint_root, adapter_name="default")

    return trainer, tokenizer


# -----------------------------
# Helpers (reduce duplication)
# -----------------------------
def _normalize_story_example(example: dict) -> dict:
    """Build a unified item for story JSONL examples."""
    return {
        "mode": "story",
        "story": example.get("story", ""),
        "noise": example.get("noise", "unknown"),
        "gold_story": example.get("gold_story", ""),
        "meta": {},
    }


def _normalize_leaf_example(example: dict) -> dict:
    """Build a unified item for LEAF essay examples."""
    title = example.get("essay_title", "").strip()
    if "essay_text" in example:
        essay = example.get("essay_text", "").strip()
    else:
        essay = example.get("essay", "").strip()
    
    # Capture all metadata from the original example
    meta = {k: v for k, v in example.items() if k not in ("essay_title", "essay_text", "essay")}
    
    return {
        "mode": "leaf",
        "story": f"Title: {title}\n{essay}", 
        "title": title,
        "essay_text": essay,
        "meta": meta,
    }


def _build_prompt(tokenizer, template: str, story: str) -> str:
    prompt_text = template.format(story=story)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_plain_prompt(template: str, story: str) -> str:
    return template.format(story=story)


def _generate_completion(trainer, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    prediction_output = trainer.model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        prediction_output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def _create_gemini_client():
    api_key = os.getenv("###")
    if not api_key:
        raise EnvironmentError(
            "### is not set. Export it before running with --backend gemini."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is not installed. Install it with: pip install google-genai"
        ) from exc

    return genai.Client(api_key=api_key)


def _create_openai_client():
    api_key = os.getenv("###")
    if not api_key:
        raise EnvironmentError(
            "### is not set. Export it before running with --backend openai."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is not installed. Install it with: pip install openai"
        ) from exc

    return OpenAI(api_key=api_key)


def _extract_gemini_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    chunks = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)

    return "\n".join(chunks).strip()


def _generate_completion_gemini(client, model_name: str, prompt: str, max_retries: int = 3) -> str:
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = _extract_gemini_text(response)
            if text:
                return text

            logger.warning("Gemini returned an empty text response.")
            return ""
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break

            sleep_seconds = 2 ** attempt
            logger.warning(
                f"Gemini request failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                f"Retrying in {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Gemini generation failed after {max_retries} attempts: {last_error}"
    ) from last_error


def _extract_openai_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    output_items = getattr(response, "output", None) or []
    chunks = []
    for item in output_items:
        content_items = getattr(item, "content", None) or []
        for content in content_items:
            content_text = getattr(content, "text", None)
            if content_text:
                chunks.append(content_text)

    return "\n".join(chunks).strip()


def _generate_completion_openai(client, model_name: str, prompt: str, max_retries: int = 3) -> str:
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model_name,
                input=prompt,
            )
            text = _extract_openai_text(response)
            if text:
                return text

            logger.warning("OpenAI response returned empty text.")
            return ""
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break

            sleep_seconds = 2 ** attempt
            logger.warning(
                f"OpenAI request failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                f"Retrying in {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"OpenAI generation failed after {max_retries} attempts: {last_error}"
    ) from last_error


def _build_output_record(item: dict, completion: str, prompt_type: str, gen_time: float) -> dict:
    if item["mode"] == "leaf":
        return {
            "title": item.get("title", ""),
            "essay_text": item.get("essay_text", ""),
            "feedback": completion,
            "metadata": {
                "generation_time": gen_time,
                "prompt_type": prompt_type,
                **item.get("meta", {}),
            },
        }
    # story mode
    else:
        return {
            "story": item.get("story", ""),
            "noise": item.get("noise", "unknown"),
            "gold_story": item.get("gold_story", ""),
            "feedback": completion,
            "metadata": {
                "generation_time": gen_time,
                "prompt_type": prompt_type,
            },
        }

def main(
    checkpoint_root: str,
    output_jsonl: str,
    use_base_model_only: bool = False,
    prompt_type: str = "feedback",
    data_file: str = str(DEFAULT_DATA_FILE),
    leaf_mode: bool = False,
    override: bool = False,
    model_name: str = "gemma-2-9b-it",
    backend: str = "hf",
    max_examples: int = None,
):
    """Generate feedback for stories or essays and save to JSONL."""
    checkpoint_root = str(checkpoint_root) if checkpoint_root else None
    data_file = str(data_file)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = backend.lower()
    if backend not in {"hf", "gemini", "openai"}:
        raise ValueError(f"Unknown backend: {backend}")

    logger.info(f"Starting feedback generation. Output file: {output_path}")
    if leaf_mode:
        logger.info("Mode: LEAF essays")
    else:
        logger.info(f"Mode: Story")
    if max_examples is not None:
        logger.info(f"Limiting run to max_examples={max_examples}")

    if backend == "hf":
        logger.info(f"Using HF model: {model_name}")
        base_model = str((TRANSFORMER_ROOT / model_name).resolve())

        model_load_start = time.time()
        trainer, tokenizer = load_generation_model(base_model, checkpoint_root, use_base_model_only)
        gemini_client, openai_client = None, None
        logger.info(f"Loaded model and tokenizer in {time.time() - model_load_start:.1f}s")
    elif backend == "gemini":
        logger.info(f"Using Gemini API model: {model_name}")
        if checkpoint_root:
            logger.info("Ignoring --checkpoint_root for Gemini backend.")

        model_load_start = time.time()
        gemini_client = _create_gemini_client()
        trainer, tokenizer, openai_client = None, None, None
        logger.info(f"Initialized Gemini client in {time.time() - model_load_start:.1f}s")
    else:
        logger.info(f"Using OpenAI API model: {model_name}")
        if checkpoint_root:
            logger.info("Ignoring --checkpoint_root for OpenAI backend.")

        model_load_start = time.time()
        openai_client = _create_openai_client()
        trainer, tokenizer, gemini_client = None, None, None
        logger.info(f"Initialized OpenAI client in {time.time() - model_load_start:.1f}s")

    # Select prompt template
    prompt_template = get_prompt_template(prompt_type)
    
    logger.info(f"Using prompt type: {prompt_type}")

    # Load dataset based on mode
    if leaf_mode:
        dataset = load_dataset("json", data_files=data_file, split="train")
        num_examples = len(dataset)
        logger.info(f"Loaded LEAF dataset: {num_examples} examples from {data_file}")
    else:
        dataset = load_dataset("json", data_files=data_file, split="train")
        num_examples = len(dataset)
        logger.info(f"Loaded story dataset: {num_examples} examples from {data_file}")

    # Check for existing output and determine starting index
    start_idx = 0
    file_mode = "w" if override else "a"
    
    if not override and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f_check:
            for line in f_check:
                if line.strip():
                    start_idx += 1
        logger.info(f"Found existing output file with {start_idx} records. Resuming from index {start_idx}.")
    elif override and output_path.exists():
        logger.info(f"Overriding existing output file: {output_path}")
    
    start_all = time.time()
    processed_examples = 0
    with open(output_path, file_mode, encoding="utf-8") as f_out:
        # Unified iterator and loop
        if leaf_mode:
            iterator = (_normalize_leaf_example(ex) for ex in dataset)
        else:
            iterator = (_normalize_story_example(ex) for ex in dataset)

        for idx, item in enumerate(iterator):
            if idx < start_idx:
                continue
            if max_examples is not None and processed_examples >= max_examples:
                logger.info(
                    f"Reached max_examples={max_examples}. Stopping early after {processed_examples} processed example(s)."
                )
                break

            example_start = time.time()
            story = item.get("story", "")
            if not story:
                logger.warning(f"Example {idx}: Empty text, skipping")
                continue

            # Prompt + generate
            if backend == "hf":
                prompt = _build_prompt(tokenizer, prompt_template, story)
                completion = _generate_completion(trainer, tokenizer, prompt)
            elif backend == "gemini":
                prompt = _build_plain_prompt(prompt_template, story)
                completion = _generate_completion_gemini(gemini_client, model_name, prompt)
            else:
                prompt = _build_plain_prompt(prompt_template, story)
                completion = _generate_completion_openai(openai_client, model_name, prompt)

            if idx < 3:
                logger.info(f"Example {idx} prompt: {prompt}")
            if idx < 3:
                logger.info(f"Example {idx} completion: {completion}")

            # Build and write record
            rec = _build_output_record(item, completion, prompt_type, time.time() - example_start)
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            processed_examples += 1

            # Progress
            if idx < 3 or (idx + 1) % 50 == 0 or (idx + 1) == num_examples:
                elapsed = time.time() - example_start
                logger.info(f"[{idx + 1}/{num_examples}] gen={elapsed:.1f}s | story='{story[:100]}...'")

    logger.info(
        f"Finished. Time={time.time() - start_all:.1f}s"
    )
    logger.info(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate feedback from a model and save to JSONL.")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "gemini", "openai"],
                        help="Backend to use: hf (local transformers), gemini (Gemini API), or openai (OpenAI API)")
    parser.add_argument("--checkpoint_root", type=str, help="Path to the checkpoint directory.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--use_base_model_only", action="store_true", help="Use the base model without LoRA.")
    parser.add_argument("--prompt_type", type=str, default="feedback_only",
                        help="Type of prompt to use (default: feedback_only)")
    parser.add_argument("--data_file", type=str, default=str(DEFAULT_DATA_FILE),
                        help="Path to the data file. For stories: JSONL.")
    parser.add_argument("--leaf_mode", action="store_true",
                        help="Process LEAF essays (essay_text) instead of stories")
    parser.add_argument("--override", action="store_true",
                        help="Override existing output file instead of appending/resuming")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on number of new examples to process in this run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-2-9b-it",
        help=(
            "Model name to use. For hf backend: local model directory under transformer-models. "
            "For gemini/openai backends: API model ID, e.g. gemini-2.5-flash or gpt-5."
        ),
    )
    args = parser.parse_args()

    if args.backend == "hf" and not args.use_base_model_only and not args.checkpoint_root:
        parser.error("--checkpoint_root is required for hf backend unless --use_base_model_only is specified.")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max_examples must be >= 1 when provided.")

    main(
        args.checkpoint_root,
        args.output_jsonl,
        args.use_base_model_only,
        args.prompt_type,
        args.data_file,
        args.leaf_mode,
        args.override,
        args.model,
        args.backend,
        args.max_examples,
    )
