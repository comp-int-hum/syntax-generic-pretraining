import argparse
import json
import tarfile

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ts_tgz", help="Targz input")
    parser.add_argument("--output", nargs="+", help="Output jsonls, len == --sets")
    parser.add_argument("--n", type=int, default=10000, help="Number of samples/set")
    parser.add_argument("--sets", type=int, default=1, help="Number of nonoverlapping datasets to draw")
    args, rest = parser.parse_known_args()

    with tarfile.open(args.ts_tgz, "r:gz") as t_i, open(args.output, "wt") as o_jl:
        c_s = 0
        for member in t_i:
            if c_s > args.n:
                break
            if member.isfile():
                print(f"Extracting data from {member.name}")
                with t_i.extractfile(member) as i_f:
                    f_d = i_f.read()
                    for i,story in enumerate(json.loads(f_d)):
                        if c_s > args.n:
                            break
                        else:
                            c_s += 1
                            o_jl.write(json.dumps({"structure": [[story["story"]]], "datafile": member.name, "i":i})+"\n")
                            
        
