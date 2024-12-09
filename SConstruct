import os
import os.path

from steamroller import Environment


vars = Variables("custom.py")
vars.AddVariables(

    # Gutenberg data
    ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    ("GUTENBERG_PATH", "", "${DATA_ROOT}/gutenberg/"),
    ("PG_CATALOG", "", "data/pg_catalog.csv"),

    # Tinystories data
    ("TS_TAR", "", "data/TinyStories_all_data.tar.gz"),
    ("TS_N", "", 5),
    ("POS_REP", "", ["NOUN", "PROPN"]),

    # SPARQL query
    ("SPARQL_QUERY","", "data/en_authors.txt"),
    
    # Filter settings
    ("P1_THRESH", "", 90), #similarity threshold for pass 1 of fuzzy matching, paried with bd_thresh
    ("P2_THRESH", "", 92), #similarity threshold for pass 2 of fuzzy matching, used alone
    ("BD_THRESH", "", 5), #allowed birthdate delta
    ("OMIT_AUTHORS","",["Herman Melville"]), #temporary measure to omit a given author, uses WD authorname
    ("MAX_WORKS","", 3), #maximum number of works per author for data balancing purposes
    ("FOLDS", "", 1),

    # SLURM settings
    ("CPU_QUEUE", "", "some_queue"),
    ("CPU_ACCOUNT", "", "some_account"),    
    ("GPU_QUEUE", "", "another_queue"),
    ("GPU_ACCOUNT", "", "another_account"),
    ("GPU_COUNT", "", 1),
    ("WORK_DIR", "", "work"),

    # Data Split settings
    ("TRAIN_PORTION", "", 0.7),
    ("DEV_PORTION", "", 0.1),
    ("TEST_PORTION", "", 0.2),

    # Random Seed
    ("RANDOM_SEED", "", 42),

    # Wandb settings
    ("USE_WANDB", "", False),
    ("WANDB_PROJECT", "", "BabyLlama_1"),

    # Training
    ("TRAINER_CONFIG_1", "", "config/gpt-705M.yaml"),
    ("TRAINER_CONFIG_2", "", "config/llama-360M.yaml"),
    ("STUDENT_CONFIG", "", "config/llama-58M.yaml")
)

env = Environment(
    variables=vars,
    BUILDERS={
        "QueryWD" : Builder(
              action="python scripts/author_gather_metadata.py --sparql ${SOURCES} --output ${TARGETS}"
	    ),
	    "GBAuthorFuzzy": Builder(
	      action="python scripts/author_gb_fuzzy.py "
	             "--input ${SOURCES} --output ${TARGETS} "
		     "--pg_catalog ${PG_CATALOG} "
		     "--author_omit ${OMIT_AUTHORS} "
		     "--p1_thresh ${P1_THRESH} --p2_thresh ${P2_THRESH} --bd_thresh ${BD_THRESH} --max_works ${MAX_WORKS} --random_state ${RANDOM_SEED}"
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
        ),
	    "TokenizeSplit" : Builder(
            action = (
                "python scripts/tokenize_split.py "
                "--input ${SOURCES[0]} "
                "--tokenizer ${SOURCES[1]} "
                "--output ${TARGETS} "
            )
        ),
        "TrainTeacher" : Builder(
            action = (
                "python scripts/train_teacher.py "
                "--train_data ${SOURCES[0]} "
                "--eval_data ${SOURCES[1]} "
                "--tokenizer_path ${SOURCES[2]} "
                "--config ${CONFIG} "
                #"--lr ${LR} "
                "--random_seed ${RANDOM_SEED} "
                "--use_wandb ${USE_WANDB} "
                "--wandb_project ${WANDB_PROJECT} "
                "--wandb_name ${WANDB_NAME} "
                "--output_dir ${TARGET}"
            )
        ),
        "DistillTrainStudent" : Builder(
            action = (
                "python scripts/distill_train_student.py "
                "--train_data ${SOURCES[0]} "
                "--eval_data ${SOURCES[1]} "
                "--tokenizer_path ${SOURCES[2]} "
                "--teacher_dir_1 ${SOURCES[3]} "
                "--teacher_dir_2 ${SOURCES[4]} "
                "--config ${CONFIG} "
                #"--lr ${LR} "
                "--random_seed ${RANDOM_SEED} "
                "--use_wandb ${USE_WANDB} "
                "--wandb_project ${WANDB_PROJECT} "
                "--wandb_name ${WANDB_NAME} "
                "--output_dir ${TARGET}"
            )
        ),
        "LoadTSData" : Builder(
            action = (
                "python scripts/load_ts.py "
                "--ts_tgz ${SOURCES} "
                "--output ${TARGETS} "
                "--n ${TS_N}"
            )
        ),
        "POSTransform" : Builder(
            action= (
                "python scripts/pos_transform.py "
                "--input ${SOURCES} "
                "--output ${TARGETS} "
                "--data_name ${DATA_NAME} "
                "--pos_rep ${POS_REP}"
                )
        )
    }
)

ts_input = env.File(env["TS_TAR"])

ts_data = env.LoadTSData(source = ts_input, target = "${WORK_DIR}/ts_subset.jsonl")

pos_data = env.POSTransform(source = ts_data, target = "${WORK_DIR}/ts_pos.jsonl", DATA_NAME = "ts")


"""
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

tokenized_train_dev_test = []
for data_split in train_dev_test:
    tokenized_train_dev_test.append(env.TokenizeSplit(
        source = [data_split, tokenizer],
        target = str(data_split) + ".pt"
))

train_data, dev_data, test_data = tokenized_train_dev_test

teacher_1 = env.TrainTeacher(
    source = [train_data, dev_data, tokenizer],
    target = Dir(f"{env['WORK_DIR']}/teacher_1"),
    CONFIG = env["TRAINER_CONFIG_1"],
    WANDB_NAME = "Teacher_1"
)

teacher_2 = env.TrainTeacher(
    source = [train_data, dev_data, tokenizer],
    target = Dir(f"{env['WORK_DIR']}/teacher_2"),
    CONFIG = env["TRAINER_CONFIG_2"],
    WANDB_NAME = "Teacher_2"
)

student = env.DistillTrainStudent(
    source = [train_data, dev_data, tokenizer, teacher_1, teacher_2],
    target = Dir(f"{env['WORK_DIR']}/student"),
    CONFIG = env["STUDENT_CONFIG"],
    WANDB_NAME = "Student"
)
"""


