import os
import os.path

from steamroller import Environment




vars = Variables("custom.py")
vars.AddVariables(

    # Experiment combination configs
    ("CONFIGS","", [
            {"data": ["ts_pos","ax"], "tokenizer": "ts_pos"},
            {"data": ["ts","ax"], "tokenizer": "ts"},
            {"data": ["ts_pos", "ts"], "tokenizer": "ts_pos"}
        ]),
    ("TOKENIZERS","", ["ts_pos","ts", "ax"]),
    ("SAVE_TOTAL", "Max number of models that can be saved in a run", 10),

    # Gutenberg data
    ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    ("GUTENBERG_PATH", "", "${DATA_ROOT}/gutenberg/"),
    ("PG_CATALOG", "", "data/pg_catalog.csv"),
    ("WORK_DIR", "", "work/"),

    # Tinystories data
    ("TS_TAR", "", "data/TinyStories_all_data.tar.gz"),
    ("TS_N", "", 7500),
    ("POS_REP", "", ["NOUN", "PROPN"]), # "NOUN PROPN"),# [repr('NOUN'), repr('PROPN')]),

    # Arxiv data
    ("AX_ZIP", "", "data/arxiv.zip"),
    ("AX_N", "", 10000),

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
    ("STEAMROLLER_ENGINE", "", "slurm"),
    ("CPU_QUEUE", "", "parallel"),
    ("CPU_ACCOUNT", "", "tlippin1"),    
    ("GPU_QUEUE", "", "a100"),
    ("GPU_ACCOUNT", "", "tlippin1_gpu"),
    ("GPU_COUNT", "", 1),
    ("GRID_MEMORY", "", "64GB"),

    # Data Split settings
    ("TRAIN_PORTION", "", 0.7),
    ("DEV_PORTION", "", 0.1),
    ("TEST_PORTION", "", 0.2),

    # Random Seed
    ("RANDOM_SEED", "", 42),

    # Wandb settings
    ("USE_WANDB", "", True),
    ("WANDB_PROJECT", "", "BabyLlama_1"),

    # Training
    # ("TRAINER_CONFIG_1", "", "config/gpt-705M.yaml"),
    # ("TRAINER_CONFIG_2", "", "config/llama-360M.yaml"),
    # ("STUDENT_CONFIG", "", "config/llama-58M.yaml")
    
    ("TRAINER_CONFIG_1", "", "config/llama-smoll-345M.yaml"),
    ("TRAINER_CONFIG_2", "", "config/llama-smoll-345M.yaml"),
    ("STUDENT_CONFIG", "", "config/llama-smoll-345M.yaml"),

    ("LOAD_FROM_MODEL", "", None)
    
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
                "--random_seed ${RANDOM_SEED} "
                "--include_preface"

            )
        ),
        "TrainTokenizer" : Builder(
            action = (
                "python scripts/train_tokenizer.py "
                "--input ${SOURCES} "
                "--output ${TARGETS} "
                "--special_tokens ${POS_REP}"
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
                "--output_dir ${TARGETS[0]} "
                "--checkpoints ${TARGETS[1]} "
                "--save_total ${SAVE_TOTAL} "
                "${LOAD_FROM_MODEL and f'--load_from_model ' + LOAD_FROM_MODEL or ''}"
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

        "LoadAXData" : Builder(
            action = (
                "python scripts/load_ax.py "
                "--ax_zip ${SOURCES} "
                "--output ${TARGETS} "
                "--n ${AX_N}"
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

def cpu_task_config(name, time_required, memory_required=env["GRID_MEMORY"]):
    return {
        "STEAMROLLER_ACCOUNT": env["CPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["CPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
    }

def gpu_task_config(name, time_required, memory_required=env["GRID_MEMORY"]):
    return {
        "STEAMROLLER_ACCOUNT": env["GPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["GPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
        "STEAMROLLER_GPU_COUNT": env["GPU_COUNT"],
    }

ts_input = env.File(env["TS_TAR"])
ax_input = env.File(env["AX_ZIP"])

ts_data = env.LoadTSData(source = ts_input,
                         target = "${WORK_DIR}/ts_subset.jsonl",
                         **cpu_task_config(f"load_ts", "6:00:00"),)
ax_data = env.LoadAXData(source = ax_input,
                         target = "${WORK_DIR}/ax_subset.jsonl",
                         **cpu_task_config(f"load_ax", "6:00:00"),)

ts_pos_data = env.POSTransform(source = ts_data,
                               target = "${WORK_DIR}/ts_subset_pos.jsonl",
                               DATA_NAME = "ts",
                               **cpu_task_config(f"words", "6:00:00"))

data = {"ts": ts_data, "ax": ax_data, "ts_pos": ts_pos_data}

splits = {}
for dname, dset in data.items():
    splits[dname] = env.TrainingSplit(
        source = dset,
		target = ["${WORK_DIR}/${DNAME}_data.train", "${WORK_DIR}/${DNAME}_data.dev", "${WORK_DIR}/${DNAME}_data.test"],
		DNAME = dname,
  		**cpu_task_config(f"training_split_{dname}", "6:00:00"),
    )


tokenizers = {}
for tname in env["TOKENIZERS"]:
    tokenizers[tname] = env.TrainTokenizer(
        source = splits[tname][0],
		target = "${WORK_DIR}/${TNAME}_tokenizer.json",
        TNAME = tname,
        **cpu_task_config(f"train_tokenizer_{tname}", "6:00:00"),
    )

first_dataset = "ts"
second_dataset = "ax"

first_tokenized = []
for split in splits[first_dataset]:
	first_tokenized.append(
     	env.TokenizeSplit(
			source = [split, tokenizers[first_dataset]],
			target = str(split) + ".pt",
			**cpu_task_config(f"ax_tokenization", "6:00:00"),
		)
    )

second_tokenized = []
for split in splits[second_dataset]:
	second_tokenized.append(
     	env.TokenizeSplit(
			source = [split, tokenizers[second_dataset]],
			target = str(split) + ".pt",
			**cpu_task_config(f"ax_tokenization", "6:00:00"),
		)
    )

teacher_1, t1_checkpoints = env.TrainTeacher(
    source = [first_tokenized[0], first_tokenized[1], tokenizers[first_dataset]],
    target = [Dir(f"{env['WORK_DIR']}/teacher_1/output"), Dir(f"{env['WORK_DIR']}/teacher_1/checkpoints")],
    CONFIG = env["TRAINER_CONFIG_1"],
    WANDB_NAME = "Teacher_1",
    SAVE_TOTAL = env["SAVE_TOTAL"],
    **gpu_task_config(f"teacher_1_ts", "12:00:00"),
)

checkpoints_1_dirs = [os.path.join(t2_checkpoints, d) for d in os.listdir(t1_checkpoints) if 'checkpoint-' in d]

for checkpoint_dir in checkpoints_1_dirs:
    checkpoint_teacher, new_checkpoint_dir = env.TrainTeacher(
        source=[second_tokenized[0], second_tokenized[1], tokenizers[second_dataset]],
        target=[Dir(os.path.join(checkpoint_dir, "output")), Dir(os.path.join(checkpoint_dir, "sub_checkpoints"))],
        CONFIG=env["TRAINER_CONFIG_1"],
        WANDB_NAME=f"{os.path.basename(checkpoint_dir)}_training",
        SAVE_TOTAL=env["SAVE_TOTAL"],
        LOAD_FROM_MODEL=Dir(checkpoint_dir),
        **gpu_task_config(f"{os.path.basename(checkpoint_dir)}_ts", "12:00:00"),
    )


teacher_2, t2_checkpoints = env.TrainTeacher(
    source = [first_tokenized[0], first_tokenized[1], tokenizers[first_dataset]],
    target = [Dir(f"{env['WORK_DIR']}/teacher_2/output"), Dir(f"{env['WORK_DIR']}/teacher_2/checkpoints")],
    CONFIG = env["TRAINER_CONFIG_2"],
    WANDB_NAME = "Teacher_2",
    SAVE_TOTAL = env["SAVE_TOTAL"],
    **gpu_task_config(f"teacher_2_ts", "12:00:00"),
)



# teacher_2 = env.TrainTeacher(
#     source = [train_data, dev_data, tokenizer],
#     target = Dir(f"{env['WORK_DIR']}/teacher_2"),
#     CONFIG = env["TRAINER_CONFIG_2"],
#     WANDB_NAME = "Teacher_2"
# )

# student = env.DistillTrainStudent(
#     source = [train_data, dev_data, tokenizer, teacher_1, teacher_2],
#     target = Dir(f"{env['WORK_DIR']}/student"),
#     CONFIG = env["STUDENT_CONFIG"],
#     WANDB_NAME = "Student"
# )


"""
train_dev_test = env.TrainingSplit(
    source = pos_data,
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


