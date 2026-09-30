# nanochat training report

Generated: 2026-08-23 18:07:44

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
- Characters: 642,247
- Lines: 14,941
- Files: 50
- Tokens (approx): 160,561
- Dependencies (uv.lock lines): 2,257

Run started: 2026-08-23 18:07:45

---

## Tokenizer training
timestamp: 2026-08-23 18:08:39

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 48.3430
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 19
- token_bytes_mean: 6.6029
- token_bytes_std: 2.8250


## Tokenizer evaluation
timestamp: 2026-08-23 18:08:46

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
timestamp: 2026-08-24 03:01:28

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
- model_tag: FULLRUN_MUON
- Number of parameters: 1,384,124,976
- Number of FLOPs per token: 4.945112e+09
- Calculated number of iterations: 2088
- Number of training tokens: 2,189,426,688
- Tokens : Scaling params ratio: 3.0000
- DDP world size: 1
- warmup_ratio: 0.0000
- warmdown_ratio: 0.5000
- final_lr_frac: 0.0000
- Minimum validation bpb: 0.8159
- Final validation bpb: 0.8159
- CORE metric estimate: 0.1772
- MFU %: 35.89%
- Total training flops: 1.082696e+19
- Total training time: 505.08m
- Peak memory usage: 54765.37MiB


## Base model evaluation
timestamp: 2026-08-24 03:41:57

- model: base_model (step 2088)
- CORE metric: 0.1779
- train bpb: 0.8160
- val bpb: 0.8156
- hellaswag_zeroshot: 0.2264
- jeopardy: 0.0548
- bigbench_qa_wikidata: 0.4204
- arc_easy: 0.4832
- arc_challenge: 0.1001
- copa: 0.2200
- commonsense_qa: 0.1216
- piqa: 0.3678
- openbook_qa: 0.1307
- lambada_openai: 0.3567
- hellaswag: 0.2230
- winograd: 0.1575
- winogrande: 0.0292
- bigbench_dyck_languages: 0.1390
- agi_eval_lsat_ar: 0.1087
- bigbench_cs_algorithms: 0.4235
- bigbench_operators: 0.1667
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.1914
- coqa: 0.1834
- boolq: -0.3673
- bigbench_language_identification: 0.1762
- sample 0: <|bos|>The capital of France is Paris. It is the largest city in the country and the most populous city in
- sample 1: <|bos|>The chemical symbol of gold is Au. It is a soft, malleable, ductile metal. It
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Saturday. If today is Saturday, then tomorrow will be Saturday. If tomorrow is
- sample 3: <|bos|>The opposite of hot is cold. The opposite of cold is warm. The opposite of hot is cold.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
- sample 5: <|bos|>My favorite color is blue. I love it. I love it. I love it. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is the number of times 5*x + 3 = 13.
If
- unconditioned 0: <|bos|>The Summary can be formatting or presentation in the format and manner for the purpose of describing and analyzing the culture gathered by a certain group of people or an experiment. Volumes Vengeance on myths losing their value and turn themselves into mythology that can stay with them until they turn into mythology lost to their hesitate. All myths advantages and disadvantages crosschecked. Literally myth is simply a story to fuel a strong emotion in you but in a one sided argument, one individual draws out their particular myth and fills it into the social imagery constructs a myth country by country in brittany pippins most contemporary publication per their interest in the
- unconditioned 1: <|bos|>New phenomenon revealed by the World Health Organization
A phenomenon that is taking place in the Caribbean.
SNOWVIGRE, the island, at an elevation of 375 feet, moves seasonally, and almost nobody is familiar with it. Fortunately, the Cayman Islands has had its fair share of both.
Extremely warm temperatures, and calm seas in general, have made it the preferred destination as far back as the 1800s. Independent engineers, biologists, sea explorers, adventurers, and adventurers have made it their home. We have witnessed three of these made the trip to the Bahamas and South Florida during
- unconditioned 2: <|bos|>Inquiry Based Learning in a Montessori Program for Kids
17/11/14 CultureClub Videos
A new culture of learning emerges
Every school child is interested and engaged in the experiences of the school. It is important for the child to know new things and practice the skills that she has learned in the program, so that she will develop the same high level of attention and commitment and hopefulness that she had in the program. In the next year of school, new skills, attitudes and behaviors will have built in, building the child's foundations for lifelong success. The Tajala program appeals to the child and the interest she has
- unconditioned 3: <|bos|>Civil war 1861, or the "Burning of Washington," essay
(co-led by editor assistant james elis years after the Civil War begins, society and the press agreed that many slaves were eager to pursue felix donohubieta on indian indians were in a consistent state of war with the kanse for and we are grateful for their contributing editor summaries editor lives: the human cost of war this article provides critical. The civil war in 1861, the end of a period of divided decision-making and some visions of humanity's future go quite deeply into the present.
War: the primary textbook
- unconditioned 4: <|bos|>Dialysis means to replace or clean the blood in the human body. Urine is produced only when the kidneys are working correctly. Normally, the kidneys hold a tiny amount of waste products, such as protein and salts. Hemodialysis cleans these waste products from the blood and cleans them back into the body via a dialysis machine. The cleaned blood is circulated through a tube, called a catheter, to the dialysis machine. The blood then flows back into the body through a large vein called the main artery.
The main artery that supplies most (78%) of the blood in the body is in your arm, usually in the back of your
- unconditioned 5: <|bos|>Freedom by association. “. . . the decision to give your child a child, a toddler, or a dog, can cause them to be exposed to a literal Marxist-Leninist world view.” Researchers from Universidade Federal do Rio Grande do Sul’s School of Social Psychology found that “in a top-down way [condo de população placement law” enacted in 2008] half of adolescents reported being exposed to oppositional view of gender and operated on a homophobic world view; and slower-response students were receiving support on a non-transmissible world view, often because of discrimination of being
- unconditioned 6: <|bos|>SANEYLNIE WAS A SECRET RESTORE FAMILY OF SKIFAD AND STONE EARLY MINUTE
GRAY by Scott Stoltzfus
It’s the Bracewell-Gissels Road Valley Grablestone-Fedora Coronagh farm that is famous around the world! As well known as spectacular cliff stones in Swindon, which can be visited with days off, or close to summer-fair for travelers, and tourist-friendly and visited regularly by tourists in the whole of England, the limestone quarry on the to the north side of the valley
- unconditioned 7: <|bos|>The bladder is located in the lower urinary tract, so it is part of a complex structure. The bladder stores urine until it leaves the kidneys, by storing it and then moving it to a place where it is expelled by the kidneys. The bladder stores urine until bacteria or other organisms (streptococcus that have been fitted with a needle) are detected or enough bacteria are detected to allow it to cause a urine infection .
Deep Vein Thrombosis (DVT)
Doctor Leigh Encke must explain the idea that Deep Venous Thrombosis is a different type of blood clot from DVT due to certain circumstances.



## Summary

- Characters: 642,247
- Lines: 14,941
- Files: 50
- Tokens (approx): 160,561
- Dependencies (uv.lock lines): 2,257

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.1779   | -        | -        |

Total wall clock time: 9h34m
