import argparse
import json
import os
from time import time

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.keywords import extract_keywords

# from src.utils import request_log

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--dataset_path", type=str, default="data")
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--test_size", type=int, default=100)
parser.add_argument("--fp32", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--keywords", action="store_true")


def main():
    args = parser.parse_args()

    current_time = int(time())
    print(f"Output file: generated_{current_time}.json")

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint if args.checkpoint else "skt/ko-gpt-trinity-1.2B-v0.5")
    tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")

    model.eval()
    model.to(args.device)

    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count())

    generated = []

    dataset = json.load(open(os.path.join(args.dataset_path, "test.json"), "r", encoding="utf-8"))

    if args.keywords:
        keywords = process_map(extract_keywords, dataset, max_workers=os.cpu_count() // 2, chunksize=32)
        keywords = list(filter(lambda x: x is not None, keywords))

        # Extract keywords and the first sentence
        dataset = list(map(lambda x: (" ".join(x[0]) + tokenizer.eos_token, x[1][0]), zip(keywords, dataset)))

    else:
        dataset = list(map(lambda x: x[0], dataset))

    dataset = dataset[:args.test_size]

    for prompt in tqdm(dataset):
        tokenized = tokenizer([prompt], return_tensors="pt", return_token_type_ids=True).to(args.device)

        with torch.no_grad(), autocast(enabled=not args.fp32):
            token_type_ids = tokenized["token_type_ids"]
            del tokenized["token_type_ids"]

            output = model.generate(
                    **tokenized,
                    penalty_alpha=0.6,
                    top_k=8,
                    max_new_tokens=args.max_new_tokens,
            )

            tokenized["labels"] = tokenized["input_ids"].clone()
            tokenized["labels"][token_type_ids == 0] = -100

            loss = model(**tokenized).loss

        decoded = tokenizer.decode(output[0])
        generated.append({
            "prompt": prompt,
            "generated": decoded,
            "loss": loss.item(),
        })

        # Logging
        # try:
        #     request_log(decoded)
        # except:
        #     # RPC not available
        #     pass

        json.dump(generated, open(f"./generated_{current_time}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
