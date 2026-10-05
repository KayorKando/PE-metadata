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

### Getting labels for a new CSV

If a CSV has text but no labels, annotate it with MAPLE's own script, `MAPLE/biorxiv/utility_eval/extract_metadata.py` (Gemini 2.5 flash-lite, MAPLE schema). It writes the labels as `map_<attribute>` columns; drop the `map_` prefix before running `run.py`.
