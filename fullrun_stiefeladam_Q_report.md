# nanochat training report

Generated: 2026-09-02 22:57:04

## Environment

### Git Information
- Branch: main
- Commit: 3282bbb (dirty)
- Message: optimizer setup functions in gpt

### Hardware
- Platform: Linux
- CPUs: 64 cores (64 logical)
- Memory: 2015.0 GB
- GPUs: 1x NVIDIA H100 80GB HBM3
- GPU Memory: 79.2 GB total
- CUDA Version: 12.8
- Hourly Rate: $3.00/hour

### Software
- Python: 3.12.0
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 642,351
- Lines: 14,944
- Files: 50
- Tokens (approx): 160,587
- Dependencies (uv.lock lines): 2,257

Run started: 2026-09-02 22:57:05

---

## Tokenizer training
timestamp: 2026-09-02 22:58:12

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 62.0190
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 19
- token_bytes_mean: 6.6029
- token_bytes_std: 2.8250


## Tokenizer evaluation
timestamp: 2026-09-02 22:58:19

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 403 | 4.51 | +0.2% |
| korean | 893 | 745 | 1.20 | 797 | 1.12 | -7.0% |
| code | 1259 | 576 | 2.19 | 620 | 2.03 | -7.6% |
| math | 1834 | 936 | 1.96 | 1025 | 1.79 | -9.5% |
| science | 1112 | 260 | 4.28 | 258 | 4.31 | +0.8% |
| fwe-train | 4208518 | 900364 | 4.67 | 892476 | 4.72 | +0.9% |
| fwe-val | 4768657 | 1027270 | 4.64 | 1023546 | 4.66 | +0.4% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 403 | 4.51 | -4.1% |
| korean | 893 | 364 | 2.45 | 797 | 1.12 | -119.0% |
| code | 1259 | 309 | 4.07 | 620 | 2.03 | -100.6% |
| math | 1834 | 832 | 2.20 | 1025 | 1.79 | -23.2% |
| science | 1112 | 249 | 4.47 | 258 | 4.31 | -3.6% |
| fwe-train | 4208518 | 874799 | 4.81 | 892476 | 4.72 | -2.0% |
| fwe-val | 4768657 | 1001442 | 4.76 | 1023546 | 4.66 | -2.2% |


## Base model training
timestamp: 2026-09-03 07:57:43

- run: dummy
- device_type: 
- fp8: True
- fp8_recipe: tensorwise
- depth: 24
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 2048
- window_pattern: SSSL
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 3.0000
- device_batch_size: 16
- total_batch_size: -1
- embedding_lr: 0.3000
- unembedding_lr: 0.0040
- weight_decay: 0.2000
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- adam_beta1: 0.8000
- adam_beta2: 0.9500
- warmup_ratio: 0.0000
- warmdown_ratio: 0.5000
- final_lr_frac: 0.0000
- resume_from_step: -1
- eval_every: 250
- eval_tokens: 20,971,520
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- save_every: -1
- model_tag: FULLRUN_STIEFELADAM_Q
- Number of parameters: 1,384,124,976
- Number of FLOPs per token: 4.945112e+09
- Calculated number of iterations: 2088
- Number of training tokens: 2,189,426,688
- Tokens : Scaling params ratio: 3.0000
- DDP world size: 1
- warmup_ratio: 0.0000
- warmdown_ratio: 0.5000
- final_lr_frac: 0.0000
- Minimum validation bpb: 0.8359
- Final validation bpb: 0.8359
- CORE metric estimate: 0.1666
- MFU %: 35.47%
- Total training flops: 1.082696e+19
- Total training time: 511.72m
- Peak memory usage: 54981.23MiB


## Base model evaluation
timestamp: 2026-09-03 08:37:32

- model: base_model (step 2088)
- CORE metric: 0.1620
- train bpb: 0.8359
- val bpb: 0.8355
- hellaswag_zeroshot: 0.1828
- jeopardy: 0.0331
- bigbench_qa_wikidata: 0.3769
- arc_easy: 0.4411
- arc_challenge: 0.0648
- copa: 0.2400
- commonsense_qa: 0.1319
- piqa: 0.3428
- openbook_qa: 0.1093
- lambada_openai: 0.3122
- hellaswag: 0.1874
- winograd: 0.2601
- winogrande: 0.0560
- bigbench_dyck_languages: 0.1420
- agi_eval_lsat_ar: 0.1087
- bigbench_cs_algorithms: 0.3955
- bigbench_operators: 0.1524
- bigbench_repeat_copy_logic: 0.0312
- squad: 0.1463
- coqa: 0.1566
- boolq: -0.4864
- bigbench_language_identification: 0.1803
- sample 0: <|bos|>The capital of France is Paris. It is the largest city in the world and the most populous city in
- sample 1: <|bos|>The chemical symbol of gold is Au. It is a soft, malleable, ductile, malle
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Saturday. If you have a computer, you can use the Internet to find the
- sample 3: <|bos|>The opposite of hot is cold. The opposite of cold is hot. The opposite of hot is cold.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
- sample 5: <|bos|>My favorite color is red. I love the color red. I love the color red. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is a prime number. If 5*x + 3 = 13,
- unconditioned 0: <|bos|>The Summary can be found here (Advanced, our second lesson for the workshop). The reminder to use the words by their Latin end names in our Pastor’s Latin.
The “N” refers to a bi-part turn — Latin “divine” means “that one being,” or womb. Regardless of the form of the neuter cross, masculine or feminine, it is feminine and feminine are always paired with an “a-ween.
THE DEMANDS OF THE NEWS
The dictatorship pressured us into protections based on the murder of journalists by journalists murdered in February. “Just closure” is often employed in insurance journalism by
- unconditioned 1: <|bos|>New York – May 24, 2017 – Approximately nine percent of the state’s workforce remains illiterate. Follow the 40,000 local projects that advocate for readability support, which direct and provide financial support to teachers implementing 40k instructional strategies. Prevent poverty and raise incomes for educators through educating and developing tutors through state and local support.
“Today, there are a great number of adults in poverty. There are adults who are very likely to be illiterate. If we didn’t support the thought absorption Through Education, Literacy Month, focusing on the three things that made the month exceptional, we would not have had the
- unconditioned 2: <|bos|>Ininking and triggering in a self inspired manner is a powerful tool to improve the learning process. Vygotsky highlighted the importance of using action as a “multiply” process when implementing the theory. It is important to realize that we know new perceptions and therefore the meaning of things and events and what they mean to us. Socially it doesn't matter how we are in the world, we know not everything we think and perceive around us has worth. This respect yourself, doesn't matter, but makes a difference.
When you are viewing events as having more importance than they actually are, you can begin realising that things
- unconditioned 3: <|bos|>Civil war over slavery
In recent years the factor of human rights has gained more and more importance in the region. The most prominent years of this weathering were 1816 and the aftermath of the First Consulate movement, which came to a close in 1858. It was this period that saw the beginning of the abolishment of distinctions under the category of race between (African, black and mulatto) and human beings.
The fight for a basic liberty of all in Africa continued in 1862 when black men in Transvaal were finally freed and fundamental emancipation was made a priority, instead of seeing paid.
During the Second World
- unconditioned 4: <|bos|>Dialysis means to replace or rehydrate the exteriors of our living organisms by extracting waste products and toxins from their cells before they exit our organism.
Dialysis therapy works on three key ideas that believe that complementarily producing new cellular components is essential for normal development and function of human body systems and organs.
These ideas are- Renormalization of proteins and micro deviants and receptors Cytokines that across a number of cell types,mates, and lyrines and improves function Cell survival by inducing cell survival Evolution (Ha3-like) and Frankenstein Cell regeneration Rods, granulocytes increase
B
- unconditioned 5: <|bos|>Freedom by association. “. . . the decision and act of organized human groups, often from time to time entailing widespread obedience to authority, and a unity in belief and action, without any distinction between individual and group.” James Madison Jr. State Disciplines Report, Knightly Edition 448; Hamilton & Madison (1882) F43-1.[liberal Political Economy]
Jefferson was a “commitment to separation of powers and protection of individual rights” and opposed Federal sovereignty when he cited “a substantial delay in the receiving and complying with of any measure of legislation, of which we have specimens.”
- unconditioned 6: <|bos|>SANE ACLETIC PROGRAMS
Continuation of line drawing where the classification locale is initially defined and initialized, using both Span and Arc
G', RS by SA after an earlier version consisting of the K' before S-G and RS in Arc.
The original K' was purposefully changed to read "geometrically equivalent to K' in full-color," and then translated (in the simplified form noted below). Thus, ae, Ø, d' used to be read as "geometric equivalents of e - CRJ" and e - CPJ to read as "electrical symbols
- unconditioned 7: <|bos|>The bladder is located in the lower part of the torso, fills during a cyclic flow through the reflex activity of radiculacupuncture more than 10, years. The function is similar to that in a kidney, the part being about reflex activity. By means of its waves under reflexes initiated and that have a specific cause, the bladder sends signal to signal a contraction of the sphincters of a certain readiness, as a rule to simply open the flow.
Normally, a bladder empty by means of » eliminate » is emptied by a healthcare oncologist. Depending on the anomaly, bleeding from the urinary tract, urinary dist


## Summary

- Characters: 642,351
- Lines: 14,944
- Files: 50
- Tokens (approx): 160,587
- Dependencies (uv.lock lines): 2,257

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.1620   | -        | -        |

Total wall clock time: 9h40m
