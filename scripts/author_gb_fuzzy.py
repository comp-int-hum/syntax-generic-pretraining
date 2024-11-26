import re
import argparse
from thefuzz import fuzz
from thefuzz import process
import json
import csv
from collections import defaultdict, Counter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Jsonl of auth control texts")
    parser.add_argument("--pg_catalog", help="PG catalog csv location")
    parser.add_argument("--output", help="Output jsonl")
    parser.add_argument("--p1_thresh", type=int, default=90, help="Pass 1 similarity threshold")
    parser.add_argument("--p2_thresh", type=int, default=92, help="Pass 2 similarity threshold")
    parser.add_argument("--bd_thresh", type=int, default=5, help="Pass 1 birthday delta threshold")
    args, rest = parser.parse_known_args()

    a_t_id = defaultdict(lambda: defaultdict(dict))
    a_to_a = {}
    print("Getting Gids for {} using {}".format(args.input, args.pg_catalog))
    with open(args.pg_catalog, "rt") as pg_i:
        pg_r = csv.DictReader(pg_i)
        for r in pg_r:
            date = re.search("[0-9]+\-[0-9]+", r["Authors"].replace("?",""))
            author = re.sub("[^a-zA-Z]+"," ",r["Authors"].replace(","," ")).strip()
            birthProp = 0000
            deathProp = 0000
            if date:
                birthProp, deathProp = date.group(0).split("-")
                birthProp = int(birthProp)
                deathProp = int(deathProp)
            a_t_id[r["Authors"]]["Works"][r["Title"]]=r["Text#"]
            a_t_id[r["Authors"]]["gb_birth"]=birthProp
            a_t_id[r["Authors"]]["gb_death"]=deathProp
            a_to_a[author]=r["Authors"]
    print("Collected {} GB authors".format(len(a_t_id)))

    #pass 1
    #meant to collect high certainty items, so high fuzzy comparison scores and low birthdate deltas
    pass_1 = []
    leftover = []
    used_gb = []
    n = 0
    n_total = 0
    with open(args.input, "rt") as s_in, open(args.output, "wt") as s_o:
        for line in s_in:
            n_total +=1
            jline = json.loads(line)
            author = jline.get("authorLabel",{}).get("value", "")
            wd_birth = int(jline.get("birthDate",{}).get("value", "0000")[0:4])
            gb_a = process.extractOne(author, a_to_a.keys())
            #gb_a = process.extractOne(author, a_t_id.keys(),scorer=fuzz.partial_ratio)
            if gb_a[1] >= args.p1_thresh:
                gb_record = a_t_id.get(a_to_a[gb_a[0]])
                if abs(gb_record["gb_birth"]-wd_birth) <= args.bd_thresh: 
                    n+=len(gb_record["Works"])
                    used_gb.append(gb_a[0])
                    pass_1.append({"gb_works": gb_record["Works"], "gb_author": a_to_a[gb_a[0]], "score":gb_a[1], "gb_birth": gb_record["gb_birth"], "assigned": "pass1"} | jline)
                    s_o.write(json.dumps({"gb_works": gb_record["Works"], "gb_author": a_to_a[gb_a[0]], "gb_birth": gb_record["gb_birth"], "assigned": "pass1"} | jline)+"\n")
                else:
                    leftover.append(jline)
            else:
                leftover.append(jline)


                    
        print("Retrieved {} Gids from {} authors of {} total WD authors in pass 1".format(n, len(pass_1), n_total))

        #TODO: deduplication, looks like its the multiple author in Gutenberg fields, but a small set
        counts = Counter([p["gb_author"] for p in pass_1])
        dups = {key:value for key, value in counts.items() if value > 1}
        for k in dups.keys():
            for p in pass_1:
                if k == p["gb_author"]:
                    print(p)
                    
        #pass 2: assign leftovers based on similarity thresh alone
        a2_to_a = {ma: ga for ma, ga in a_to_a.items() if ma not in used_gb}

        print("Leftover WD from pass 1: {} Leftover GB from pass1 {}".format(len(leftover), len(a2_to_a)))
        
        n_pass2 = 0
        na_pass2 = 0
        for line2 in leftover:
            author = line2.get("authorLabel",{}).get("value", "")
            
            gb_a = process.extractOne(author, a2_to_a.keys())
            if gb_a[1] >= args.p2_thresh:
                gb_record = a_t_id.get(a2_to_a[gb_a[0]])
                na_pass2 += 1
                n_pass2+=len(gb_record["Works"])
                s_o.write(json.dumps({"gb_works": gb_record["Works"], "gb_author": a2_to_a[gb_a[0]], "gb_birth": gb_record["gb_birth"], "assigned": "pass2"} | line2)+"\n")

        print("Retrieved {} additional Gids from {} additional authors in pass 2".format(n_pass2, na_pass2))
        print("Total Works: {} Total Authors {}".format(n+n_pass2, len(pass_1)+na_pass2))

