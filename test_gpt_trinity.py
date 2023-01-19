import argparse

import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

# from src.utils import request_log

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--fp32", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda")


def main():
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
    model.eval()
    model.to(args.device)

    if args.device == "cpu":
        torch.set_num_threads(24)

    while True:
        prompt = input("Prompt > ")
        tokenized = tokenizer(prompt, return_tensors="pt").to(args.device)

        with torch.no_grad(), autocast(enabled=not args.fp32):
            output = model.generate(
                    **tokenized,
                    penalty_alpha=0.6,
                    top_k=8,
                    max_new_tokens=256,
            )

        decoded = tokenizer.decode(output[0])
        print(f"Output > {decoded}")

        # Logging
        # try:
        #     request_log(decoded)
        # except:
        #     # RPC not available
        #     print("* Logging failed")
        #     pass


if __name__ == '__main__':
    main()
