import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "Jsonl of works, authors, etc.")
    parser.add_argument("--output", help = "Jsonl containing retrieved texts")
    args, _ = parser.parse_known_args()
    
    with open(args.input, "r") as input_file, open(args.output, "wt") as output_file:
        for line in input_file:
            author_info = json.loads(line)
            extracted_works = []
            for name, text_num in author_info["gb_works"].items():
                text_path = os.path.join(
                    args.gutenberg_path,
                    *[c for c in text_num[:-1]],
                    text_num,
                    text_num + "-h",
                    text_num + "-h.htm"
                )
                if os.path.exists(text_path):
                    with open(text_path, "rt", errors="ignore") as text_file:
                        content = text_file.read()
                        extracted_works.append(
                            {
                                "name": name,
                                "id": id,
                                "raw": content
                            }
                        )
            author_info["extracted_works"] = extracted_works
            output_file.write(json.dumps(author_info) + "\n")
                        