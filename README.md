# PE-metadata

Metadata drift in Private Evolution (follow-up to [MAPLE](https://github.com/elichien-google/MAPLE)).

## Experiment 1: k-NN blindness vs. linear probe

**Question.** For each metadata attribute, does the vote embedding φ see it?
We measure this in two ways, using the same random splits so the two scores can be compared directly:

| Score | What it measures | How |
|---|---|---|
| **k-NN visibility** | what PE's vote can *use* | Split the data into two disjoint halves (ref / query). For each query point, take its k = 5 nearest ref points (L2, like MAPLE's vote) and compute the share with the same label. No majority vote, so there are no ties. |
| **Probe visibility** | what φ *encodes* at all | Fit a logistic regression on ref, test it on query. |

Both scores are chance-corrected to a kappa scale: **0 = chance, 1 = perfect**. Blindness = 1 − kappa.
For k-NN, the chance level is $\sum_c p_c^2$, the agreement of a random pair. For the probe, the score is Cohen's kappa.

**How to read the plot** (`knn_vs_probe.png`):

| k-NN | Probe | Meaning | What quota can do |
|---|---|---|---|
| high | high | visible | not needed |
| **low** | **high** | φ encodes it, but other directions dominate the distance | **quota should help a lot** |
| low | low | φ does not encode it | quota fixes the distribution, not within-cell ranking |

The `--top` (default 3) attributes with the highest k-NN blindness are marked `selected`. These are the quota attributes.

### Setup

- φ = `sentence-t5-base` with `max_seq_length = 1024`, as in MAPLE (`PE.py --emb_name`, `--emb_dim`).
- Data: MAPLE's `biorxiv/biorxiv_train_metadata.csv` (Gemini-2.5-flash-lite labels), subsampled to n = 2000.
- Labels are mapped onto the MAPLE schema: exact match, then prefix match, otherwise `Other` (or `Not Specified / Not Applicable` for `sample_size`).
- `word_count` uses MAPLE's binning, `round(words / 50) * 50`.
- 20 repeats of random half/half splits; the table reports mean ± sd.

### Run on the cluster

`/home` is for personal files only; large regenerable data goes in `/data`.
Keep the code in `/home`, and put the MAPLE data, the Hugging Face model cache, the embedding cache and the results in `/data`:

```bash
export PE_DATA=/data/$USER/PE-metadata          # embedding cache + results
export HF_HOME=/data/$USER/hf_cache              # sentence-t5-base weights
mkdir -p $PE_DATA $HF_HOME
git clone https://github.com/elichien-google/MAPLE.git $PE_DATA/MAPLE

# virtual environment, kept in /data (not /home)
python -m venv /data/$USER/envs/pe-metadata
source /data/$USER/envs/pe-metadata/bin/activate

# RTX Pro 6000 (Blackwell) needs a CUDA 12.8+ build of PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

for n in 2000 5000 0; do                         # 0 = all 28,846 rows
  CUDA_VISIBLE_DEVICES=0 python run.py --csv $PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv \
    --n $n --batch_size 256 --out $PE_DATA/results/n$n
done
```

Running three n values checks whether the top-3 ranking depends on sample size: k-NN agreement rises as points get denser.
At full n the slow part is the probe, which runs on CPU; add `--repeats 5` there.

### Run (local)

```bash
python -m venv .venv                                               # virtual environment
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/elichien-google/MAPLE.git ../MAPLE   # data
python -m tests.test_toy                                           # sanity check, ~10 s
python run.py --csv ../MAPLE/biorxiv/biorxiv_train_metadata.csv --n 2000
```

Embedding 2k abstracts takes a few minutes on CPU or Apple MPS. Embeddings are cached in `$PE_DATA/cache/` (default `./cache/`).

Outputs in `$PE_DATA/results/` (or `--out`):

- `summary.csv`: per-attribute scores and the selected attributes
- `per_repeat.csv`: raw scores for every split
- `knn_vs_probe.png`: the plot described above
- `config.json`: the run settings, plus the Spearman ρ between the two rankings

Other options:

- `--k`, `--repeats`, `--top`, `--model` change the corresponding settings.
- `--attributes` runs a subset of attributes, or any extra CSV column.
- `--csv` accepts any CSV with a `text` column, for example a round-0 synthetic pool.

### Privacy note

This run uses private records (the bioRxiv train set) with their annotated labels, so it is a **non-private diagnostic**.
In the actual method, the blind attributes are chosen from the **round-0 synthetic pool** with inherited AIM labels, which is post-processing and has no privacy cost.
Run the same script on that pool:

```bash
python run.py --csv path/to/PE_iter_0.csv --n 0
```

### Rerun on the round-0 synthetic pool

This is the setting the real method uses: pick the quota attributes on the round-0 pool, which only depends on AIM-privatized metadata and public in-context examples (post-processing, no extra privacy cost).

`gen_round0.py` reuses MAPLE's own prompt and `RANDOM_API` and saves the pool **before** the vote.
MAPLE's `PE_iter_0.csv` is after the 1-NN vote on private data and top-K selection, so it would touch private data and is already filtered by φ.
Defaults follow MAPLE: Qwen2.5-7B-Instruct, 10 of 50 in-context examples per prompt, 2000 × 7 = 14,000 texts, AIM ε = 4.

**One command** (sets up the MAPLE env if missing, generates one shard per GPU in parallel, merges, then runs the blindness analysis; logs in `$PE_DATA/round0/eps4.0/logs/`):

```bash
tmux new -d -s round0 "bash ~/PE-metadata/run_round0.sh"     # options: EPS=2.0 GPUS=1,2 bash run_round0.sh
```

The steps it runs are listed below for reference.

Labels in the pool are the **requested** (AIM) values. `run.py` recomputes `word_count` from the generated text; the requested length is kept as `word_count_requested`.

```bash
# MAPLE's environment (vLLM + private-evolution), separate from pe-metadata
python3 -m venv /data/$USER/envs/maple
source /data/$USER/envs/maple/bin/activate
pip install -U pip
pip install vllm==0.10.1.1            # PyPI wheel is built for CUDA 12.8 (needed for Blackwell)
pip install "private-evolution[text] @ git+https://github.com/microsoft/DPSDA.git" datasets==4.0.0
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"   # expect +cu128, (12, 0)

# generate: one shard per GPU (Qwen2.5-7B weights, ~15 GB, go to $HF_HOME)
cd ~/PE-metadata
CUDA_VISIBLE_DEVICES=1 python gen_round0.py --maple_dir $PE_DATA/MAPLE --eps 4.0 --shard 0 --num_shards 2   # tmux window 1
CUDA_VISIBLE_DEVICES=2 python gen_round0.py --maple_dir $PE_DATA/MAPLE --eps 4.0 --shard 1 --num_shards 2   # tmux window 2
python gen_round0.py --maple_dir $PE_DATA/MAPLE --eps 4.0 --merge --num_shards 2   # -> $PE_DATA/round0/eps4.0/pool.csv

# measure blindness on the pool (pe-metadata env)
source /data/$USER/envs/pe-metadata/bin/activate
CUDA_VISIBLE_DEVICES=1 python run.py --csv $PE_DATA/round0/eps4.0/pool.csv --n 0 --repeats 5 \
  --attributes primary_research_area model_organism experimental_approach dominant_data_type \
               research_focus_scale disease_mention sample_size research_goal word_count word_count_requested \
  --batch_size 256 --out $PE_DATA/results/round0_eps4.0
```

### Getting labels for a new CSV

If a CSV has text but no labels, annotate it with MAPLE's own script, `MAPLE/biorxiv/utility_eval/extract_metadata.py` (Gemini 2.5 flash-lite, MAPLE schema). It writes the labels as `map_<attribute>` columns; drop the `map_` prefix before running `run.py`.

### Why is ρ(k-NN, probe) lower on the round-0 pool? (diagnostics)

Three checks, cheapest first:

```bash
export PE_DATA=/data/$USER/PE-metadata HF_HOME=/data/$USER/hf_cache
source /data/$USER/envs/pe-metadata/bin/activate
POOL=$PE_DATA/round0/eps4.0/pool.csv
ATTRS="primary_research_area model_organism experimental_approach dominant_data_type \
       research_focus_scale disease_mention sample_size research_goal word_count word_count_requested"

# 1. Is 0.95 vs 0.64 a real difference over 8 attributes? Which attributes move? (CPU, seconds)
python compare_rho.py --a $PE_DATA/results/n0/summary.csv --b $PE_DATA/results/round0_eps4.0/summary.csv \
    --label_a private --label_b pool

# 2. Twins: same random split as before, plus twin_rate; then a group split.
CUDA_VISIBLE_DEVICES=1 python run.py --csv $POOL --n 0 --repeats 5 --attributes $ATTRS --batch_size 256 \
    --group_cols requested --exclude_from_rho word_count word_count_requested \
    --out $PE_DATA/results/round0_eps4.0_twinrate
CUDA_VISIBLE_DEVICES=1 python run.py --csv $POOL --n 0 --repeats 5 --attributes $ATTRS --batch_size 256 \
    --group_cols requested --group_split --exclude_from_rho word_count word_count_requested \
    --out $PE_DATA/results/round0_eps4.0_groupsplit
python compare_rho.py --a $PE_DATA/results/n0/summary.csv --b $PE_DATA/results/round0_eps4.0_groupsplit/summary.csv \
    --label_a private --label_b pool_groupsplit

# 3. Are the attributes less correlated with each other in the pool? (CPU)
python label_mi.py --private $PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv --pool $POOL \
    --out $PE_DATA/results/label_mi
```

## Experiment 1b: does the vote correct each attribute?

Experiment 1 measures a proxy (5-NN agreement). Blindness alone does not create drift: a vote that ignores an attribute picks at random with respect to it and keeps its distribution. What blindness removes is the pull back toward the private distribution when the generator's pool is off.
`vote_correction.py` runs MAPLE's real vote (each private point votes for its nearest pool point, optional Gaussian noise, keep the top K = 2000) on the round-0 pool and reports, per attribute, how much of the gap to the private distribution the vote closes:

`correction = (JSD_random − JSD_vote) / (JSD_random − JSD_floor)`: 0 = no better than a blind random pick, 1 = as good as a perfect sample of size K, NaN = nothing to correct.

Noise multipliers default to MAPLE's ε = ∞ / 4 / 2 / 1 values. Embeddings come from the Experiment 1 cache.

```bash
CUDA_VISIBLE_DEVICES=1 python vote_correction.py \
  --private_csv $PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv \
  --pool_csv $PE_DATA/round0/eps4.0/pool.csv \
  --exp1_summary $PE_DATA/results/round0_eps4.0/summary.csv \
  --out $PE_DATA/results/vote_correction_eps4.0
```

This uses private data, so it is a non-private diagnostic.

