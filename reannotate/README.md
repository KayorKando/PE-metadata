# Re-annotating experiment

Can a small classifier replace an LLM annotator for the metadata labels we need in every PE round?
(Labels are needed because the generator does not follow the requested AIM metadata, so requested labels cannot be trusted.)

1. **Round-0 pool**: Qwen2.5-7B-Instruct + AIM metadata (`gen_round0.py`; reused if `$PE_DATA/round0/eps4.0/pool.csv` exists).
2. **Qwen as the annotator** (`annotate_qwen.py`): vLLM, MAPLE's own annotation prompt, output constrained to the MAPLE schema by guided JSON decoding, greedy. Labels the whole pool, and a sample of private texts (default 3,000) to compare Qwen with the Gemini labels that ship with MAPLE.
3. **One classifier per attribute** (`train_classifier.py`), trained on Qwen labels of the pool only (no private data in training), split 70/15/15 by prompt group, no class weighting:
   - `tfidf`: TF-IDF 1–2-grams + logistic regression (C chosen on val), CPU
   - `scibert`: `allenai/scibert_scivocab_uncased`, fine-tuned (3 epochs, lr 2e-5, best epoch on val)
   - `biomedbert`: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, same recipe

   - MLPs on frozen embeddings, two hidden layers (Linear d→256, ReLU, dropout 0.2, Linear 256→256, ReLU, dropout 0.2), AdamW lr 1e-3, wd 1e-4, batch 256, early stopping on val κ (patience 10, ≤ 100 epochs), in two shapes:
     - `*_mlp`: one MLP per attribute
     - `*_mh`: one shared trunk + one linear head per attribute giving that attribute's logits, trained jointly (sum of cross-entropies, missing labels ignored), early stopping on the mean val κ

     on two embeddings, **reported as separate families**:
     - `phi_*`: the vote embedding φ (sentence-t5-base, max_seq_length 1024, Experiment 1 cache). Cheapest annotator for the method (φ is computed for the vote anyway).
     - `biomed_*`: frozen, mean-pooled BiomedBERT. Tells whether fine-tuning is needed.

   The report also gives, per attribute, the κ gap "best text-encoder model − best φ model": small gap = the attribute is in φ and the L2 vote just does not use it; large gap = φ itself lacks it.

   Evaluation is always against Qwen / Gemini labels, never against any classifier's own labels.
4. **Report** (`report.py` → `REPORT.md`), per attribute × model:
   1. agreement with labels: Cohen's κ, macro-F1, accuracy
   2. marginal: JSD(predicted marginal, label marginal), the same after confusion-matrix correction (`p_true ≈ argmin‖Cᵀp − p_pred‖, p ≥ 0`, C from the val split), and a resampling floor
   3. transfer, on three evaluation sets: `pool_test_vs_qwen` (in-domain), `private_sample_vs_qwen` (real text, same annotator → text-domain gap), `private_vs_gemini` (real text, other annotator → total gap). Table A adds Qwen-vs-Gemini agreement on the same private texts (annotator gap) and generator compliance (requested AIM label vs Qwen label).

   Plus training / prediction time per model.

Private texts are used only for evaluation: a non-private diagnostic.

```bash
tmux new -d -s reann "export PE_DATA=$PE_DATA HF_HOME=$HF_HOME; bash ~/PE-metadata/reannotate/run_reannotate.sh"
tail -f $PE_DATA/reannotate/eps4.0/logs/*.log
cat $PE_DATA/reannotate/eps4.0/results/REPORT.md
```

Settings: `EPS`, `GPUS` (default `1,2`), `PRIVATE_SAMPLE` (3000), `QWEN` (annotator model). Finished steps are skipped on a rerun.
