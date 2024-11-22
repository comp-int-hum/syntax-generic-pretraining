import argparse
from wikidata.client import Client
from SPARQLWrapper import SPARQLWrapper, JSON

#
# This script does *nothing* except print out its arguments and touch any files
# specified as outputs (thus fulfilling a build system's requirements for
# success).
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sparql", help="Sparql plaintext input")
    parser.add_argument("--output", help="Output jsonl")
    args, rest = parser.parse_known_args()

    client = Client()

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    print("Gathering metadata {} from {}".format(args.output, args.sparql))
    with open(args.sparql, "rt") as s_in:
        q = s_in.read()
        sparql.setQuery(q)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        

    for r in results["results"]["bindings"]:
        id = r["item"]["value"].split("/")[-1]
        entity = client.get(id, load=True)
        ws = entity.data.get("sitelinks",{}).get("enwikisource",None)
        if ws:
            print(ws)
        
    
        
#https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&props=sitelinks&ids=Q174596&sitefilter=enwikisource        
