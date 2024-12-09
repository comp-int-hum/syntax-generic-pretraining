import argparse
import json
import tarfile
import spacy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Jsonl input")
    parser.add_argument("--output", help="Output jsonl")
    parser.add_argument("--data_name", default="ts", help="Number of samples")
    parser.add_argument("--pos_rep", nargs="+", default=[], help="Pos tags to replace")
    args, rest = parser.parse_known_args()

    nlp = spacy.load("en_core_web_sm")
    print(f"Replacing {' '.join(args.pos_rep)} with generics")
    n_sents = 0
    with open(args.input, "rt") as t_i, open(args.output, "wt") as p_o:
        for line in t_i:
            j_line = json.loads(line)
            doc = nlp(j_line["story"])
            story = {"sents": [], "tokens": [], "datafile": j_line["datafile"], "i": j_line["i"]}
            for sent in doc.sents:
                sent_toks = []
                initial_offset = doc[sent.start].idx
                running_offset = 0
                sent_end = doc[sent.end].idx if sent.end < len(doc) else doc[-1].idx + 1
                orig_sent = j_line['story'][initial_offset:sent_end]
                for token in sent:
                    if token.pos_ in args.pos_rep:
                        sent_toks.append(token.text)
                        pos_mask = "<"+token.pos_+">"
                        orig_sent = orig_sent[:(token.idx-initial_offset+running_offset)] + pos_mask + orig_sent[(token.idx-initial_offset+running_offset+len(token.text)):]
                        running_offset += len(pos_mask) - len(token.text)
                story["sents"].append(orig_sent)
                story["tokens"].append(sent_toks)
            n_sents += len(story["sents"])
            p_o.write(json.dumps(story)+"\n")
    print(f"Processed {n_sents} sentences") 

        
