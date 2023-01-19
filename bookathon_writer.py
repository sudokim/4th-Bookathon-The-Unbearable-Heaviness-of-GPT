import argparse
import json
import os
import re
import time

import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.keywords import extract_keywords

# from src.utils import request_log

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--draft", type=str, default=None, help="Path to draft directory")
parser.add_argument("--max_length", type=int, default=384, help="Maximum length of generated text")
parser.add_argument("--fp32", action="store_true", default=False, help="Use 32-bit floating point precision")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
parser.add_argument("--num_last_sentences", type=int, default=2, help="Number of last sentences to use as prompt")
parser.add_argument("--penalty_alpha", type=float, default=0.6, help="Penalty alpha for contrastive search")


def main():
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")

    model.eval()
    model.to(args.device)

    if args.device == "cpu":
        torch.set_num_threads(24)

    # Create new file for the entire document
    if args.draft is None:
        # Create from new draft
        draft_directory = f"draft/{int(time.time())}"
        os.makedirs(draft_directory, exist_ok=True)

        print(f"Created new draft: {draft_directory}")
    else:
        # Create from existing draft
        draft_directory = f"./draft/{args.draft}"

        print(f"Loaded draft: {draft_directory}")

    draft_file_path = os.path.join(draft_directory, "draft.txt")
    draft_history_path = os.path.join(draft_directory, "history.json")
    draft_log_path = os.path.join(draft_directory, "log.txt")

    print("Writing to", draft_directory)
    print("- Draft:", draft_file_path)
    print("- Version history:", draft_history_path)
    print("- Log:", draft_log_path)

    # Data structure for version history
    if os.path.exists(draft_history_path):
        # Existing draft - load history
        draft_history = json.load(open(draft_history_path, "r"))
    else:
        # New draft - create history
        draft_history: list[dict[str, str]] = []
    draft_history: list[dict[str, str]]
    draft: str

    # Log file
    log_file = open(draft_log_path, "a+")

    # Get first draft
    if os.path.exists(draft_file_path):
        initial_keyword = None
        draft = ""
    else:
        # New draft - create file
        draft_file = open(draft_file_path, "w")
        draft = input("Initial prompt > ").strip()
        draft_file.write(draft)
        draft_file.close()

        initial_keyword = input("Initial keyword separated in \" \" > ").split()

    sentence_window = args.num_last_sentences
    while True:
        # Get keywords from prompt
        if initial_keyword is not None:
            keywords = initial_keyword
            initial_keyword = None
        else:
            draft_file = open(draft_file_path, "r")
            draft = draft_file.read().strip()
            draft_file.close()

            keywords = extract_keywords(draft[:1_000])
            print(f"Keywords: {' '.join(keywords)}")

            additional_keywords = input("Enter extra keywords > ").strip()
            if additional_keywords != "":
                additional_keywords = additional_keywords.split()

            for additional_keyword in additional_keywords:
                if additional_keyword in keywords:
                    keywords.remove(additional_keyword)
                else:
                    keywords.append(additional_keyword)

        draft_split = re.split(r"(?<=[.!?])\s+", draft)
        last_sentences = [""] + draft_split[-sentence_window:]

        input_prompt = " ".join(keywords) + tokenizer.eos_token + " ".join(last_sentences)

        # Generate sentence
        tokenized = tokenizer(input_prompt, return_tensors="pt").to(args.device)
        with torch.no_grad(), autocast(enabled=not args.fp32):
            output = model.generate(
                    **tokenized,
                    max_length=args.max_length,
                    penalty_alpha=args.penalty_alpha,
                    top_k=8,
            )
        decoded = tokenizer.decode(output[0]).strip()
        # Remove prompt
        decoded = decoded[len(input_prompt):]

        # Ditch last sentence
        decoded = decoded.rstrip()

        if decoded[-1] in (".", "?", "!"):
            decoded = " ".join(re.split(r"(?<=[.!?])\s+", decoded)[:-1])

        generated_time = time.time()
        generated_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(generated_time))

        total_draft = draft + decoded

        result = {
            "generated_time": int(generated_time),
            "generated_time_formatted": generated_time_formatted,
            "draft": draft,
            "keywords": " ".join(keywords),
            "prompt": input_prompt,
            "generated": decoded,
            "total_draft": total_draft
        }

        # Save version history
        draft_history.append(result)

        # Write version history to file
        json.dump(draft_history, open(draft_history_path, "w"), indent=2, ensure_ascii=False)

        # Write generated sentence to draft file
        draft_file = open(draft_file_path, "w")
        draft_file.write(total_draft)
        draft_file.close()

        # Log
        # print("Log result:", request_log(input_prompt + decoded))

        # Write to log file
        log_string = f"Time: {generated_time_formatted}\nPrompt: {input_prompt}\nGenerated: {decoded}"
        log_file.write(log_string + "\n\n")
        log_file.flush()

        # Print result
        print(log_string)

        # Word count
        print(f"Characters: {len(total_draft)}, Words: {len(total_draft.split())}", end="\n\n")

        # Wait for user input
        print(f"Enter the number of previous sentences to use. "
              f"(Default: {args.num_last_sentences}) Type 'exit' to exit.")
        user_input = input().strip()
        if user_input == "exit":
            break
        try:
            sentence_window = int(user_input)
        except ValueError:
            sentence_window = args.num_last_sentences

    log_file.close()


if __name__ == "__main__":
    main()
