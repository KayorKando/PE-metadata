// ---------- minimal dark theme (plain Typst, no packages) ----------
#let mut = luma(133)
#let acc = rgb("#73e2e6")
#let deck-title = state("deck-title", "")

#let alert(body) = text(fill: acc, body)

#let setup(short-title: "", body) = {
  deck-title.update(short-title)
  set page(
    width: 16cm, height: 9cm,   // same as beamer 16:9
    fill: black,
    margin: (x: 0.9cm, top: 0.7cm, bottom: 0.9cm),
    footer: context {
      set text(size: 7pt, fill: mut)
      if counter(page).get().first() > 1 [
        #deck-title.get() #h(1fr) #counter(page).display()
      ]
    },
  )
  set text(font: ("New Computer Modern Sans", "Noto Sans", "Helvetica Neue", "Noto Sans CJK TC", "PingFang TC"), size: 11pt, fill: white, lang: "en")
  set par(justify: false, leading: 0.65em, spacing: 0.9em)
  set list(marker: ([#text(fill: mut, size: 0.8em)[•]], [#text(fill: mut)[–]]), spacing: 0.7em)
  set enum(numbering: n => text(fill: mut)[#n.])
  show math.equation: set text(font: "New Computer Modern Math")
  show strong: set text(fill: white)
  body
}

#let title-slide(title: "", subtitle: "", author: "", date: "") = page(footer: none)[
  #v(1fr)
  #text(size: 17pt, weight: "bold")[#title]
  #v(0.5em)
  #text(size: 11pt, fill: mut)[#subtitle]
  #v(1.4em)
  #text(size: 9pt, fill: mut)[#author]
  #v(0.2em)
  #text(size: 8pt, fill: mut)[#date]
  #v(1fr)
]

#let slide(title, body) = page[
  #text(size: 13pt, weight: "bold")[#title]
  #v(-0.5em)
  #line(length: 100%, stroke: 0.4pt + mut.transparentize(55%))
  #v(0.3em)
  #body
]
// ---------- end theme ----------

// ---------- deck helpers ----------
#let note(body) = text(size: 7.5pt, fill: mut, body)
#let dense(body) = { set text(size: 8.6pt); set par(leading: 0.5em, spacing: 0.62em); set list(spacing: 0.5em); set enum(spacing: 0.5em); body }
#let tbl(..args) = {
  set text(size: 8.5pt)
  table(stroke: none, inset: (x: 4pt, y: 2.6pt),
    table.hline(stroke: 0.5pt + mut),
    ..args,
    table.hline(stroke: 0.5pt + mut))
}
#let hd(body) = text(fill: mut, body)

#show: setup.with(short-title: "Metadata drift in Private Evolution · 2026-11")

#title-slide(
  title: "Metadata Drift in Private Evolution",
  subtitle: "Quota sweep, fine-tuned vote embedding, next experiments, and a model of drift",
  author: "Kando Hsieh (謝濟遠) · Eli Lab",
  date: "Advisor meeting · November 2026",
)

// ===================== setup =====================
#slide[Setup: the vote can only correct attributes that $phi$ separates][
  #dense[
  MAPLE on bioRxiv, $epsilon = 4$: $T = 4$ rounds, $K = 2000$ survivors, $L = 6$ variations per survivor.
  Each private text votes for its nearest candidate under $phi$ (sentence-t5). The vote histogram gets $cal(N)(0, sigma^2)$ noise. The top-$K$ candidates survive.

  Drift has two steps:
  + The LLM does not follow the requested metadata. For dominant\_data\_type, compliance $kappa approx 0.56$.
  + The vote cannot correct an attribute that $phi$ does not separate. The error stays and grows over rounds.

  #alert[Blind set] $B$: the 3 attributes with the lowest k-NN same-label agreement under $phi$ on the round-0 pool: word\_count, dominant\_data\_type, sample\_size.

  Metrics on the $K$ survivors of each round:
  - JSD per attribute against the real test set. Qwen gives the labels.
  - MAUVE against the real test set (2,000 texts).
  ]
]

// ===================== Part 1: results =====================
#slide[Drift reproduces on Nano5, and length causes most of it][
  #dense[
  #align(center, image("figs/drift_t5.png", height: 4.3cm))
  #v(-0.3em)
  t5 MAPLE ($lambda = 0$), mean of 5 seeds. Blind mean JSD: 0.182 (round 0) $->$ 0.283 (round 4). Other attributes stay near 0.11. MAUVE increases from 0.59 to 0.72.

  #alert[word\_count] causes most of the increase: 0.21 $->$ 0.49. dominant\_data\_type increases less (0.16 $->$ 0.22). sample\_size does not increase (0.17 $->$ 0.14).
  ]
]

#slide[Quota under t5: lower blind JSD, no significant MAUVE gain][
  #dense[
  Each round, the quota sets the survivor count per cell $c$ of $B$:
  $K_c prop hat(P)(c)^lambda dot R_t (c)^(1 - lambda)$.
  $R_t$ is the cell mix of the top-$K$ of the vote. $hat(P)$ comes from the AIM records, so it costs no extra $epsilon$.

  #align(center, image("figs/quota_t5.png", height: 3.05cm))
  #v(-0.4em)
  Paired by seed against $lambda = 0$. Last round, $n = 5$, 95% CI.
  - Blind JSD decreases monotonically: $-0.049$ ($lambda = 0.25$) to $-0.153$ ($lambda = 1$). All CIs exclude 0.
  - MAUVE at $lambda = 0.5$: $+0.045 plus.minus 0.051$. The CI includes 0.
  - sample\_size gets worse as $lambda$ increases (0.14 $->$ 0.22). $hat(P)$ uses AIM/Gemini labels. The evaluation uses Qwen labels.
  ]
]

#slide[A fine-tuned vote embedding separates the content attributes][
  #dense[
  #grid(columns: (1fr, 1fr), gutter: 0.8em,
    image("figs/knn_kappa.png", height: 5.9cm),
    [
      #v(0.3em)
      BiomedBERT with one #alert[normalized-softmax] head per content attribute (Zhai & Wu, BMVC 2019).

      Training data: the round-0 synthetic pool with Qwen labels. No private text enters training, so the embedding costs no extra $epsilon$.

      Mean k-NN $kappa$ of the 8 content attributes: 0.418 (t5) $->$ 0.582 (ft).

      The embedding keeps most of the t5 structure: recall\@10 of t5 neighbours is 0.263 (frozen: 0.302).

      word\_count is not a training target. Its $kappa$ stays near 0 for all three encoders.
    ],
  )
  ]
]

#slide[ft vs t5 ($n = 5$): blind JSD $-0.20$, no significant MAUVE change][
  #dense[
  #align(center, image("figs/ft_vs_t5.png", height: 4.2cm))
  #v(-0.3em)
  The seed pairs the two arms (vote noise and vLLM sampling). Last round, 95% CI.
  - $lambda = 0$: blind JSD #alert[$-0.197 plus.minus 0.011$], other JSD $-0.039 plus.minus 0.006$, MAUVE $-0.035 plus.minus 0.054$.
  - Under ft, blind JSD decreases in every round (0.185 $->$ 0.086). Under t5, it increases after round 1.
  #note[Thin lines: single seeds. The local copies of ft seeds 2–4 stop at round 3. The ft round-4 point is the $n = 5$ mean from Nano5.]
  ]
]

#slide[MAUVE–JSD frontier: ft alone beats every t5 point on JSD][
  #dense[
  #align(center, image("figs/frontier.png", height: 4.45cm))
  #v(-0.3em)
  Last round. Point labels give $lambda$. Error bars: sd over 5 seeds (t5).
  - t5: $lambda$ trades blind JSD (0.283 $->$ 0.130) against MAUVE. MAUVE peaks at $lambda = 0.5$.
  - ft at $lambda = 0$ has lower blind and other JSD than every t5 point. Its MAUVE is inside the t5 range. ft at $lambda = 1$ is worse on both axes.
  #note[ft at $lambda in {0, 0.5}$: $n = 5$ means from Nano5. ft at $lambda = 1$: seeds 0–1. MAUVE uses t5 features, so the y-axis favours t5.]
  ]
]

#slide[ft + quota adds little, and two ft results need an explanation][
  #dense[
  #grid(columns: (1fr, 1.15fr), gutter: 1em,
    [
      #tbl(columns: 4, align: (left, right, right, right),
        hd[ft, last round], hd[$lambda = 0$], hd[$0.5$], hd[$1$],
        table.hline(stroke: 0.3pt + mut),
        [word\_count], [0.113], [0.071], [0.061],
        [dominant\_data\_type], [0.070], [0.036], [0.095],
        [sample\_size], [0.076], [0.153], [0.196],
      )
      #note[Per-attribute values: seeds 0–1.]
    ],
    [
      + #alert[Quota on top of ft.] $lambda = 0.5$ halves the dominant\_data\_type error, but sample\_size error doubles. The blind mean does not change.
      + #alert[MAUVE at $lambda = 0.5$.] ft $-$ t5 $= -0.053 plus.minus 0.046$ ($n = 5$). The CI excludes 0. MAUVE uses sentence-t5 features, the vote space of the t5 arm.
      + #alert[Length.] Under ft, word\_count JSD is 0.11 (t5: 0.49). But ft does not separate length ($kappa = 0.03$). Hypothesis: length correlates with attributes that ft separates. Check: survivor length per round, and length agreement of ft neighbours.
    ],
  )
  ]
]

// ===================== Part 2: next steps =====================
#slide[Next 1: $lambda$ schedule sweep at matched mean strength][
  #dense[
  Two observations under t5 at constant $lambda = 0.5$:
  - Blind JSD still increases after round 1 (0.141 $->$ 0.181).
  - The quota moves more texts in each round: 337 in round 1, 528 in round 4. The drift pressure increases.

  The linear schedule $lambda_t = lambda t \/ T$ lost to constant $lambda$ on the lab cluster (10/17). But its mean strength is $lambda \/ 2$, so that comparison was not fair.

  New design: compare schedules at the same mean $overline(lambda) = 1/(T+1) sum_t lambda_t$.
  #tbl(columns: 3, align: (left, left, left),
    hd[schedule], hd[$lambda_t$, $t = 0..4$], hd[question],
    table.hline(stroke: 0.3pt + mut),
    [constant], [$overline(lambda)$], [baseline (runs exist)],
    [ramp], [$2 overline(lambda) t \/ T$], [Does late pressure need late correction?],
    [step], [$0$ for $t < 2$, then $5/3 overline(lambda)$], [Is correction necessary in rounds 0–1?],
    [decay], [$2 overline(lambda) (1 - t \/ T)$], [Is early correction enough?],
  )
  $overline(lambda) in {0.25, 0.5}$, t5 vote, 5 seeds: 30 new runs, about 5 jobs.
  Prerequisite: map $hat(P)$ for sample\_size into Qwen label space, or report sample\_size separately.
  ]
]

#slide[Next 2: MAUVE has the same blind spot as the t5 vote][
  #dense[
  Facts from `quota_pe.py`:
  - The t5 vote and MAUVE use the same encoder: sentence-t5-base, max length 1024.
  - The t5 arm reuses the vote vectors for MAUVE (`Xs = Xc[sel]`).
  - The references are different: the private train set for the vote, the held-out test set for MAUVE. There is no data leak.

  MAUVE quantizes these features with k-means and compares the bin histograms. If t5 does not separate an attribute, a shift in that attribute does not change the bins. #alert[MAUVE cannot penalize drift on a blind attribute.] Rounds 1 $->$ 4 under t5: word\_count JSD 0.17 $->$ 0.49, MAUVE 0.75 $->$ 0.72.

  Tests, cheapest first:
  + #alert[Controlled shift], no PE run. Split the test set into halves. Resample one half to shift one attribute at several strengths. Record JSD and t5-MAUVE. Shift each blind attribute, and disease\_mention as a control.
  + Recompute MAUVE on the saved survivors with neutral encoders (SPECTER2, e5). Check if the ft vs t5 order changes.
  + Track the correlation of mean vote count and MAUVE over rounds.
  ]
]

#slide[Next 3: drift in vanilla Aug-PE, and a survey of PE methods][
  #dense[
  Claim to test: drift comes from voting in one embedding, not from the MAPLE metadata prompts.

  #alert[Aug-PE run.] Aug-PE puts no metadata in the prompt. Label its survivors with Qwen. Measure JSD per attribute per round against the real test set. Prediction: the same blind set drifts, because the vote space is the same.

  #alert[Survey.] For each PE variant, record three items: the vote space, the use of attribute conditioning, and the reported metrics.
  #tbl(columns: 4, align: (left, left, left, left),
    hd[method], hd[vote space], hd[conditions on attributes], hd[reports attribute fidelity],
    table.hline(stroke: 0.3pt + mut),
    [Aug-PE (Xie et al., ICML 2024)], [sentence-t5], [no], [no (MAUVE, downstream)],
    [MAPLE (2026)], [sentence-t5], [DP metadata], [JSD (Gemini labels)],
    [PrE-Text (Hou et al., ICML 2024)], [to check], [to check], [to check],
    [POPri (Hou et al., ICML 2025)], [to check], [to check], [to check],
    [SecPE (2025)], [to check], [to check], [NTP on rare words],
  )
  #note[Struct-Bench (2025): in vanilla PE, KNN-Recall stays near 0.35 while KNN-Precision increases. This agrees with our pattern.]
  ]
]

// ===================== Part 3: downstream =====================
#slide[Downstream 1: tail-class classification on the real test set][
  #dense[
  Goal: show a loss that MAUVE does not see. Drift removes rare classes first, so tail recall must decrease.

  #tbl(columns: 2, align: (left, left),
    hd[item], hd[choice],
    table.hline(stroke: 0.3pt + mut),
    [target], [one content attribute at a time (Qwen labels)],
    [tail classes], [smallest classes that sum to $<= 20%$ of the real mass, each with $>= m$ real records],
    [classifier], [TF-IDF + logistic regression: no pretrained prior, fast],
    [train set], [the $K$ survivors of the last round, with Qwen labels],
    [test set], [real held-out test set],
    [metric], [macro recall on tail classes and on head classes, reported apart],
    [arms], [t5 MAPLE, t5 + quota, ft, ft + quota],
    [bounds], [upper: 2,000 real texts. Lower: the round-0 pool],
  )

  The $>= m$ rule keeps classes with support from many records. DP is not supposed to recover single-record content.

  Prediction: t5 MAPLE has the lowest tail recall. Head recall is similar for all arms.
  ]
]

#slide[Downstream 2: next-token loss on real tail-label texts][
  #dense[
  Fine-tune one small causal LM (e.g. GPT-2 small) per arm and training set. Evaluate on real texts only.
  $ cal(L)_"tail" = - 1 / N_"tok" sum_(x in D^"real"_"tail") sum_i log p_theta (x_i | x_(< i)) $
  $D^"real"_"tail"$: real test texts with a tail label (tail as in task 1). $cal(L)_"head"$ on head-label texts is the control.

  #tbl(columns: 3, align: (left, left, left),
    hd[train set], hd[what it measures], hd[note],
    table.hline(stroke: 0.3pt + mut),
    [(A) all $K$ survivors], [what the full set teaches about the tail], [same size for all arms],
    [(B) tail-label survivors], [amount + quality of the tail], [drift makes it smaller],
    [(C) as (B), token-matched], [quality of the tail only], [equal token count],
  )

  If t5 MAPLE loses on (B) but not on (C), drift removes tail texts. If it also loses on (C), the remaining tail texts are off-topic.
  References: a real train subset of the same size, and the round-0 pool. #note[Precedent: SecPE (next-word prediction on rare-word texts).]
  ]
]

#slide[Downstream 3: label-free content tail, semantic and lexical][
  #dense[
  Tasks 1–2 depend on the label schema. These two metrics do not.

  #alert[Semantic tail] (NDB, Richardson & Weiss 2018).
  - Fit k-means ($K = 100$) on real private embeddings only. Use an independent encoder: not sentence-t5, not BiomedBERT.
  - Assign each synthetic text to the nearest centroid.
  - Tail bins: the smallest bins that hold $<= 10%$ of the real mass. Report the survival ratio (synthetic share / real share) and the number of extinct bins.

  #alert[Lexical tail].
  - Word-level document frequency on real data. Keep only words with $"df" >= 5$.
  - Rare-word recall per df bin, on token-matched subsamples (length drift changes the token count).

  Difference from MAUVE: MAUVE fits its bins on real $union$ synthetic data in t5 space and gives one number. NDB fixes the bins on real content and reports the tail bins separately.

  Predictions: t5 MAPLE loses the most tail bins and rare words. ft recovers most of them. The quota improves label JSD more than the content tail.
  ]
]

// ===================== Part 4: model =====================
#slide[A model of drift with two numbers per attribute: $g$ and $rho$][
  #dense[
  #grid(columns: (1.05fr, 1fr), gutter: 0.8em,
    [
      One attribute. $p$: private distribution. $pi$: what the LLM writes without a constraint. $q_t$: candidates. $s_t$: survivors.

      *Vote.* With probability $g$, a private text with value $a$ votes for a candidate with value $a$. Otherwise it votes for a random candidate:
      $ s_t = g p + (1 - g) q_t $
      *Variation.* The LLM keeps the value with probability $rho$. Otherwise it samples from $pi$:
      $ q_(t+1) = rho s_t + (1 - rho) pi $
    ],
    [
      Together: $q_(t+1) = rho (1-g) q_t + rho g p + (1 - rho) pi$. The contraction factor per round is $rho (1 - g)$. At the fixed point,
      $ norm(s^* - p) = ((1-g)(1-rho)) / (1 - rho + rho g) norm(pi - p) $
      #align(center, image("figs/model_toy.png", height: 3.15cm))
    ],
  )
  #v(-0.5em)
  Predictions: $g = 0$ gives $s^* = pi$. A higher $g$ puts $s^*$ nearer to $p$. A lower $rho$ gives more drift. The error increases if $q_0$ is nearer to $p$ than $q^*$, and decreases otherwise. One round-0 pool can thus give a rising error under t5 and a falling error under ft.
  ]
]

#slide[Testing the model: measure $g$, $rho$, $pi$, then predict JSD per round][
  #dense[
  #tbl(columns: 3, align: (left, left, left),
    hd[quantity], hd[measurement], hd[cost],
    table.hline(stroke: 0.3pt + mut),
    [$g_k (phi)$], [chance-corrected k-NN $kappa$ on the round-0 pool], [no PE run],
    [$rho_k$], [generate from requested metadata, label with Qwen, count matches], [once, all $phi$],
    [$pi_k$], [generate with no metadata (or random metadata), label with Qwen], [once],
    [$p_k$], [Qwen labels of the private set], [available],
  )

  #alert[$phi$ sweep] at $lambda = 0$, 3 seeds each: sentence-t5, frozen BiomedBERT, ft (runs exist), plus TF-IDF, SPECTER2 and one general encoder (e5 or gte). 12 new runs, 2 jobs.

  Two tests:
  + Within one attribute, the drift rank across $phi$ must follow the rank of $g$. This excludes attribute difficulty as an explanation.
  + Put the measured $g$, $rho$, $pi$ and the observed $q_0$ into the recurrence. Compare the predicted JSD per round with the observed JSD.

  Known gaps: chance correction of $g$. DP noise lowers the effective $g$. Correlated attributes (word\_count under ft). Top-$K$ selection is not linear.
  ]
]

// ===================== close =====================
#slide[Summary and decisions][
  #dense[
  - Drift reproduces on 5 seeds. Length causes most of it.
  - Under t5, the quota lowers blind JSD monotonically in $lambda$. Its MAUVE gain is not significant.
  - The ft vote embedding lowers blind JSD by 0.20 at no extra privacy cost.
  - A model with two attribute-level numbers, $g$ and $rho$, predicts where drift goes. The $phi$ sweep tests it.

  #v(0.3em)
  #alert[Decisions for today]
  + Venue: ICML (late January) or NeurIPS (May)? ICML needs the $phi$ sweep, the Aug-PE run, one downstream task, and a second dataset.
  + Main method: ft vote embedding, with the quota as a complement?
  + Settings for ft were chosen with checks on private texts. Do we state this use, or repeat the choices on public data?
  + Run order: MAUVE controlled shift and neutral MAUVE $->$ $phi$ sweep $->$ downstream tasks $->$ schedule sweep $->$ Aug-PE.
  ]
]
