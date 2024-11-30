import os
import os.path

from steamroller import Environment


# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file.
#
# Note how, since we expect some of our build rules may want to use GPUs and/or run on
# a grid, we include a few variables along those lines that can then be overridden in
# the "custom.py" file and then used when a build rule (like "env.TrainModel") is invoked.
# Adding some indirection like this allows us finer-grained control using "custom.py",
# i.e. without having to directly edit this file.
vars = Variables("custom.py")
vars.AddVariables(
    ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    ("GUTENBERG_PATH", "", "${DATA_ROOT}/gutenberg/"),
    ("SPARQL_QUERY","", "data/en_authors.txt"),
    ("PG_CATALOG", "", "data/pg_catalog.csv"),
    ("P1_THRESH", "", 90), #similarity threshold for pass 1 of fuzzy matching, paried with bd_thresh
    ("P2_THRESH", "", 92), #similarity threshold for pass 2 of fuzzy matching, used alone
    ("BD_THRESH", "", 5), #allowed birthdate delta
    ("OMIT_AUTHORS","",["Herman Melville"]), #temporary measure to omit a given author, uses WD authorname
    ("FOLDS", "", 1),
    ("CPU_QUEUE", "", "some_queue"),
    ("CPU_ACCOUNT", "", "some_account"),    
    ("GPU_QUEUE", "", "another_queue"),
    ("GPU_ACCOUNT", "", "another_account"),
    ("GPU_COUNT", "", 1),
    ("WORK_DIR", "", "work"),
    ("TRAIN_PORTION", "", 0.7),
    ("DEV_PORTION", "", 0.1),
    ("TEST_PORTION", "", 0.2),
    ("RANDOM_SEED", "", 42),
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    # ENV = os.environ,
    # Defining a bunch of builders (none of these do anything except "touch" their targets,
    # as you can see in the dummy.py script).  Consider in particular the "TrainModel" builder,
    # which interpolates two variables beyond the standard SOURCES/TARGETS: PARAMETER_VALUE
    # and MODEL_TYPE.  When we invoke the TrainModel builder (see below), we'll need to pass
    # in values for these (note that e.g. the existence of a MODEL_TYPES variable above doesn't
    # automatically populate MODEL_TYPE, we'll do this with for-loops).
    BUILDERS={

        "QueryWD" : Builder(
              action="python scripts/author_gather_metadata.py --sparql ${SOURCES} --output ${TARGETS}"
	 ),
	"GBAuthorFuzzy": Builder(
	      action="python scripts/author_gb_fuzz.py "
	             "--input ${SOURCES} --output ${TARGETS} "
		     "--pg_catalog ${PG_CATALOG} "
		     "--author_omit ${OMIT_AUTHORS} "
		     "--p1_thresh ${P1_THRESH} --p2_thresh ${P2_THRESH} --bd_thresh ${BD_THRESH}"
        ),

        "ExtractAuthorWorksFromPG" : Builder(
			action = (
       			"python scripts/extract_author_works_from_gutenberg.py "
				"--input ${SOURCES} "
				"--gutenberg_path ${GUTENBERG_PATH} "
				"--output ${TARGETS}"
			)
   
		),
        "ExtractDocStructures" : Builder(
			action = (
				"python scripts/extract_doc_structures.py "
				"--input ${SOURCES} "
				"--output ${TARGETS}"
			)
		),
        "TrainingSplit" : Builder(
            action = (
                "python scripts/train_test_val.py "
                "--input ${SOURCES} "
                "--output_train ${TARGETS[0]} "
                "--output_dev ${TARGETS[1]} "
                "--output_test ${TARGETS[2]} "
                "--train_portion ${TRAIN_PORTION} "
                "--dev_portion ${DEV_PORTION} "
                "--test_portion ${TEST_PORTION} "
                "--random_seed ${RANDOM_SEED}"


            )
        ),
        "TrainTokenizer" : Builder(
            action = (
                "python scripts/train_tokenizer.py "
                "--input ${SOURCES} "
                "--output ${TARGETS}"
            )
        )
    }
)


input = env.File(env["SPARQL_QUERY"])

query_res = env.QueryWD(source = input, target = "${WORK_DIR}/author_query.jsonl")

gb_authors = env.GBAuthorFuzzy(source = query_res, target = "${WORK_DIR}/gb_authors.jsonl")

authors_and_extracted_works = env.ExtractAuthorWorksFromPG(
    source = gb_authors,
    target = "${WORK_DIR}/authors_and_extracted_works.jsonl"
)

extracted_structures = env.ExtractDocStructures(
	source = authors_and_extracted_works,
	target = "${WORK_DIR}/extracted_structures.jsonl"
)

train_dev_test = env.TrainingSplit(
    source = extracted_structures,
    target = ["${WORK_DIR}/data.train", "${WORK_DIR}/data.dev", "${WORK_DIR}/data.test"]
)

tokenizer = env.TrainTokenizer(
    source = train_dev_test[0],
    target = "${WORK_DIR}/tokenizer.json"
)

