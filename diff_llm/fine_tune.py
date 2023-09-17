"""Fine-tune a pre-trained LLM on diff-completion task.

NOTE:
- Consider using Seq2Seq data collator instead of LM data collator.
"""

import math
import os
import typing

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from diff_llm.data_loader import get_dataset


os.environ["WANDB_PROJECT"] = "diff-llm"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class DiffLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        out = super().compute_loss(model, inputs, return_outputs)
        return out


def fine_tune(
    model_path: str,
    data_dir: str,
    output_dir: str,
    checkpoint_dir: typing.Optional[str] = None,
    num_epochs: int = 20,
    batch_size: int = 8,
    test_size: float = 0.01,
    model_max_length: int = 1024,
    seed: int = 41,
    report_to: str = "none",
    gradient_accumulation_steps: int = 8,
    padding: str = "right",
    **kwargs,
):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side=padding,
        **kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **kwargs,
    )

    def tokenize(examples):
        import ipdb; ipdb.set_trace()
        examples["output"]
        return tokenizer(examples['example'])
    
    def filter_length(example):
        return len(example["input_ids"]) <= model_max_length + 1

    dataset = (
        get_dataset(data_dir, remove_other_columns=False)
        .map(tokenize, batched=True)
    )
    print(f"Dataset size before token length filtering: {len(dataset)}")
    dataset = dataset.filter(filter_length)
    print(f"Dataset size after token length filterinsg: {len(dataset)}")
    dataset_splits = dataset.train_test_split(test_size=test_size, seed=seed)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # TODO: DataCollatorForLanguageModeling need to do -100 masking out of the
    # target tokens that aren't part of the <AFTER>...</AFTER> sequence.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=3e-6,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=0,
        num_train_epochs=num_epochs,
        logging_steps=1,
        optim="adamw_torch",
        report_to=report_to,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=checkpoint_dir)
    eval_results = trainer.evaluate(eval_dataset=dataset_splits["test"])
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, default="EleutherAI/pythia-70m")
    parser.add_argument("--data-dir", type=str, required=True, default="datasets/diff_corpus_xs")
    parser.add_argument("--output-dir", type=str, required=True, default="models/diff_model_xs")
    parser.add_argument("--checkpoint-dir", type=str, required=False, default=None)
    parser.add_argument("--num-epochs", type=int, required=False, default=20)
    parser.add_argument("--batch-size", type=int, required=False, default=8)
    parser.add_argument("--test-size", type=float, required=False, default=0.01)
    parser.add_argument("--report-to", type=str, required=False, default="none")
    parser.add_argument("--max-length", type=int, required=False, default=1048)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=False, default=8)
    parser.add_argument("--padding", type=str, required=False, default="right")
    args = parser.parse_args()

    print(f"Arguments: {args}")

    fine_tune(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        report_to=args.report_to,
        model_max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        padding=args.padding,
    )
