"""Diff LLM inference script."""

import json
import typing

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Pipeline,
    StoppingCriteria,
)


PROMPT_TEMPLATE = """
<TITLE>
{title}
</TITLE>

<BEFORE>
{before}
</BEFORE>

<REVISION_PROMPT>
{revision_comment}
</REVISION_PROMPT>

<AFTER>
""".strip()


class DiffLLMStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequences: typing.List[str]):
        self.eos_sequences = eos_sequences

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        for eos_sequence in self.eos_sequences:
            last_ids = input_ids[:, -len(eos_sequence):].tolist()
            if eos_sequence in last_ids:
                return True
        return False


def get_inference_pipeline(
    model_path: str,
    tokenizer_path: typing.Optional[str] = None,
    model_max_length: int = 512,
) -> Pipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=model_max_length,
        padding_side="right",
    )
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return pipeline(task="text-generation", model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False, default="models/diff_model_medium")
    parser.add_argument("--tokenizer_path", type=str, required=False, default="EleutherAI/pythia-70m")
    args = parser.parse_args()

    inference_pipeline = get_inference_pipeline(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
    )

    with open(args.prompt_file, "r") as f:
        prompt_data = json.load(f)
        after = prompt_data.pop("after")
        prompt_data.pop("diff_text")

    prompt = PROMPT_TEMPLATE.format(**prompt_data)
    stopping_criteria = DiffLLMStoppingCriteria(
        eos_sequences=[
            inference_pipeline.tokenizer.encode("</after>\n"),
            inference_pipeline.tokenizer.encode("}}</after>"),
            inference_pipeline.tokenizer.encode("</after>"),
        ],
    )
    response = inference_pipeline(
        prompt,
        max_length=128,
        pad_token_id=inference_pipeline.tokenizer.eos_token_id,
        return_full_text=False,
        stopping_criteria=[stopping_criteria],
    )
    print(prompt)
    print(response[0]["generated_text"])
