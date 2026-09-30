# nanochat training report

Generated: 2026-08-24 07:43:49

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
- Characters: 642,348
- Lines: 14,943
- Files: 50
- Tokens (approx): 160,587
- Dependencies (uv.lock lines): 2,257

Run started: 2026-08-24 07:43:49

---

## Tokenizer training
timestamp: 2026-08-24 07:44:45

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 47.7702
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 19
- token_bytes_mean: 6.6029
- token_bytes_std: 2.8250


## Tokenizer evaluation
timestamp: 2026-08-24 07:44:52

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
timestamp: 2026-08-24 16:50:49

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
- model_tag: FULLRUN_STIEFELADAM
- Number of parameters: 1,384,124,976
- Number of FLOPs per token: 4.945112e+09
- Calculated number of iterations: 2088
- Number of training tokens: 2,189,426,688
- Tokens : Scaling params ratio: 3.0000
- DDP world size: 1
- warmup_ratio: 0.0000
- warmdown_ratio: 0.5000
- final_lr_frac: 0.0000
- Minimum validation bpb: 0.8908
- Final validation bpb: 0.8908
- CORE metric estimate: 0.1028
- MFU %: 34.97%
- Total training flops: 1.082696e+19
- Total training time: 518.29m
- Peak memory usage: 55412.95MiB


## Base model evaluation
timestamp: 2026-08-24 17:30:05

- model: base_model (step 2088)
- CORE metric: 0.0993
- train bpb: 0.8911
- val bpb: 0.8904
- hellaswag_zeroshot: 0.1115
- jeopardy: 0.0052
- bigbench_qa_wikidata: 0.2013
- arc_easy: 0.3507
- arc_challenge: 0.0137
- copa: 0.2400
- commonsense_qa: 0.0141
- piqa: 0.2851
- openbook_qa: 0.0693
- lambada_openai: 0.2604
- hellaswag: 0.1023
- winograd: 0.1355
- winogrande: 0.0560
- bigbench_dyck_languages: 0.0350
- agi_eval_lsat_ar: 0.1033
- bigbench_cs_algorithms: 0.3621
- bigbench_operators: 0.0952
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.0303
- coqa: 0.0720
- boolq: -0.5355
- bigbench_language_identification: 0.1759
- sample 0: <|bos|>The capital of France is Paris. It is the capital of the country. It is the capital of the
- sample 1: <|bos|>The chemical symbol of gold is gold. It is a chemical element with the symbol Au. It is a sil
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Saturday. If today was Sunday, then tomorrow will be Sunday. If today was
- sample 3: <|bos|>The opposite of hot is cold. It’s a term that’s been used to describe the state of being
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
- sample 5: <|bos|>My favorite color is red. It’s a very popular color in the world of art. It’s
- sample 6: <|bos|>If 5*x + 3 = 13, then x is 13. If 5*x + 3 = 13, then
- unconditioned 0: <|bos|>The Summary in Flight
On Monday, August 10, 2004 more than 100 hijacked by hijacked souls on American flights coasted to and from land in Morocco. In losing their first plane turn-the flight—already by night—the hijacked hijacked over nine Muslim passengers and a hijacked plane hijacked a Paris flight. That terrorist attack inspired dozens of other hijacked passengers to join the hijackers.
Today, why have more than 80 hijacked?
The hijacked hijackers themselves are not murder or genocide by journalists murdered in their neighborhoods. They’re most likely Americans who didn’t face their
- unconditioned 1: <|bos|>New York Times, April 11, 2019
Parents at The Mother Jones Movie Theatre turn a spinning topbreak into a spinning tower...
by Melissa Hardiman and Allison Moetz, MD
Working a movie in a 1950s still camera would define the bobwhite cinematic sense of the 20th century. MGM‘s reputation as a creature of the film no longer invoked. Images of rogues being roaming Gig Harbor, presidential inaugural ballots, and movie joys only grew louder with the turn of the century, made the turn a reality, and opened a whole new
- unconditioned 2: <|bos|>Inquiry Based Projects in the ZAPI Bem17
32 Theme Culture Ecology Society of BCP
Population of Amber Ramsdorf v’s Societies
Study population is the best way to tackle heady issues. The cloned zygoorophilia P02
pattern shows how new plantations of naturally antagonistic Japanese yam hybrids are cropping up in the hope of rewarding coexistence between Aryan and European populations. This area combined with respect to reservoir states has important penal habitats built in the present unhealthy wetland landscape. Legislation created in Sweden in 1959 (1986) – amending the
- unconditioned 3: <|bos|>Civil Partnership 100: What is the factor structure of Nato essay
Nato’s theory of the fundamental law of nature assumes a certain inductive law or ‘causative law’ to include the following paragraphs. First, a VA case or case consists of ‘disputed reasons’ in order to prove the Law of Fundamental Nature for the plaintiff. (4, 5, 8, 10, 11,12,13,14, 18, 20, 21, 22, 23. The Casino Civil Partnership represented a conservative, dialectical viewpoint.
Warren Derksen
- unconditioned 4: <|bos|>Dialysis means to remove or remove fluid from the exteriors of our mouths.
|Home > Nutrition Sit School Nutrition Services >||National Nutrition Examination Board - Sri Lanka||Outcomes of Diets and Diets||Basic Needs of Food Plan||Budget and Profit
Highest Situation
Population – 938,581 (2001)
Describe – The first institution – eating humans across Australia.
|Location1||Rum Packed Food||City Buying Hospital|
Assist in the following routine tasks
|Focus on Food||Yes||Respectively different Foods|
|Food
- unconditioned 5: <|bos|>Freedom of the press consists of the right of the people to read their press materials, and from time to time the right of the press to political, social, and cultural freedom. Contact figures maintain authority but abuse basic civil liberties to intimidate and intimidate individual’s elected officials, critics, and other claimants without their consent or consent. Release figures can also manipulate other law enforcement personnel to intimidate or identify them, manipulation of election records, and exoneration of defendants who avoid being publicly punished.
The federal government employs substantial governmental secrecy laws to provide a mechanism for media advocacy. The press often operates in secrecy but
- unconditioned 6: <|bos|>SANEUM, VA -- As scientists from all across the globe work to establish where emerging technologies are headed, these tests and their data become more of an indicator of the health of marine species by allowing us to better communicate currents and utilize the data collected by satellites and ocean-faring buoys to understand ecology and to track disease threats.
The ocean floor serves as a biological barrier within the petroleum industry and also as a crucial element in protecting Earth's climate. Given a strong commitment to exploration, we are used to seeing up-close the ocean, with half a million miles of ocean floor extending from the seafloor to the deep ocean plume. We
- unconditioned 7: <|bos|>The bladder is located in the groin, which is located in the lower abdominal aorta. It is located inside the abdomen. It surrounds the bladder by a pair of thin sacs, which are about ¾ of an inch long, about the width of a small human fist (Fig 1 and that in Fig 2). The bladder is contained within the aorta just beside the renal artery (Fig 3). Here you can see a block of 23rd October 1996, by Jerry M. Kurbuchen, MD, of the Western Cape General Hospital (WoA Gia&D), 31 years


## Summary

- Characters: 642,348
- Lines: 14,943
- Files: 50
- Tokens (approx): 160,587
- Dependencies (uv.lock lines): 2,257

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.0993   | -        | -        |

Total wall clock time: 9h46m
