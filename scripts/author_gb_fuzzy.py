import re
import argparse
from thefuzz import fuzz
from thefuzz import process
import json
import csv
from collections import defaultdict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Jsonl of auth control texts")
    parser.add_argument("--pg_catalog", help="PG catalog csv location")
    parser.add_argument("--output", help="Output jsonl")
    args, rest = parser.parse_known_args()

    a_t_id = defaultdict(dict)
    a_to_a = {}
    print("Getting Gids for {} using {}".format(args.input, args.pg_catalog))
    with open(args.pg_catalog, "rt") as pg_i:
        pg_r = csv.DictReader(pg_i)
        for r in pg_r:
            author = re.sub("[^a-zA-Z]+"," ",r["Authors"].replace(","," ")).strip()
            a_t_id[r["Authors"]][r["Title"]]=r["Text#"]
            a_to_a[author]=r["Authors"]
    

    n = 0
    with open(args.input, "rt") as s_in, open(args.output, "wt") as s_o:
        for line in s_in:
            jline = json.loads(line)
            author = jline.get("authorLabel",{}).get("value", "")
            gb_a = process.extractOne(author, a_to_a.keys())
            #gb_a = process.extractOne(author, a_t_id.keys(),scorer=fuzz.partial_ratio)
            if gb_a[1] >=90:
                print(gb_a[0])
                gb_works = a_t_id.get(a_to_a[gb_a[0]])
                n+=len(gb_works)
                print(gb_works)

                s_o.write(json.dumps({"gb_works": gb_works, "gb_author": a_to_a[gb_a[0]]} | jline)+"\n")                                   
    print("Retrieved {} Gids".format(n))
