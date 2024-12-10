import argparse
import json
from zipfile import ZipFile

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ax_zip", help="Zip input")
    parser.add_argument("--output", help="Output jsonl")
    parser.add_argument("--n", type=int, default=10000, help="Number of samples")
    args, rest = parser.parse_known_args()

with ZipFile(args.ax_zip) as z_i, open(args.output, "wt") as o_jl:
    c_s = 0
    for sfile in z_i.namelist():
        if c_s > args.n:
            break
        if sfile.endswith(".json"):
            print(f"Extracting data from {sfile}")
            with z_i.open(sfile, "r") as i_f:
                for line in i_f:
                    if c_s > args.n:
                        break
                    c_s += 1
                    j_line = json.loads(line)
                    o_jl.write(json.dumps({"structure": [[j_line["abstract"]]], "datafile": sfile, "i": j_line["id"]})+"\n")

