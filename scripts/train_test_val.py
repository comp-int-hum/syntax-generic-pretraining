import argparse
import json
import random
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Jsonl the structured documents")
    parser.add_argument("--train_portion", help="float of the training portion", default=0.7, type=float)
    parser.add_argument("--dev_portion", help="float of the dev portion", default=0.10, type=float)
    parser.add_argument("--test_portion", help="float of the test portion", default=0.20, type=float)
    parser.add_argument("--output_train", help="output train file", default="data.train")
    parser.add_argument("--output_dev", help="output dev file", default="data.dev")
    parser.add_argument("--output_test", help="output test", default="data.test")
    parser.add_argument("--random_seed", help="random seed", default=42, type=int)
    parser.add_argument("--include_preface", help="Include the first chapter (preface)", action="store_true", default=False)
    args, rest = parser.parse_known_args()

    # setup
    random.seed(args.random_seed)
    logging.basicConfig(level=logging.INFO)


    all_sentences = []

    with open(args.input, "rt") as fin:
        for line in fin:
            jline = json.loads(line)
            # each line is a book, with book["structure"] holding a list of chapters, with a list of paragraphs, with a list of sentences
            # the first line is a preface usually so to be safe we skip it
            for i, chapter in enumerate(jline["structure"]):
                if i == 0 and args.include_preface == False:
                    continue
                for paragraph in chapter:
                    all_sentences.extend(paragraph)

    logging.info(f"Total sentences: {len(all_sentences)}")

    random.shuffle(all_sentences)

    train_size = int(len(all_sentences) * args.train_portion)
    dev_size = int(len(all_sentences) * args.dev_portion)
    test_size = len(all_sentences) - train_size - dev_size

    logging.info(f"Train size: {train_size}")
    logging.info(f"Dev size: {dev_size}")
    logging.info(f"Test size: {test_size}")

    with open(args.output_train, "wt") as fout:
        for sentence in all_sentences[:train_size]:
            fout.write(sentence + " ")
    
    with open(args.output_dev, "wt") as fout:
        for sentence in all_sentences[train_size:train_size+dev_size]:
            fout.write(sentence + " ")

    with open(args.output_test, "wt") as fout:
        for sentence in all_sentences[train_size+dev_size:]:
            fout.write(sentence + " ")
    
    logging.info("Done")
    

    
