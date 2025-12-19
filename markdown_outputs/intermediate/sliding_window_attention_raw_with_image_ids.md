## **Sliding Window Attention Adaptation**

Yijiong Yu [a], Jiale Liu [b], Qingyun Wu [b], Huazheng Wang [a], and Ji Pei [c]


aOregon State University, {yuyiji, huazheng.wang}@oregonstate.edu
bPenn State University, {jiale.liu, qingyun.wu}@psu.edu
cDeepSolution, research@deepsolution.chat



**Abstract**


The self-attention mechanism in Transformer
based Large Language Models (LLMs) scales
quadratically with input length, making longcontext inference expensive. Sliding window attention (SWA) reduces this cost to linear complexity, but naively enabling complete
SWA at inference-time for models pretrained
with full attention (FA) causes severe longcontext performance degradation due to training–inference mismatch. This makes us wonder: _Can FA-pretrained LLMs be well adapted_
_to SWA without pretraining?_ We investigate
this by proposing Sliding Window Attention
Adaptation (SWAA), a set of practical recipes
that combine five methods for better adaptation: (i) applying SWA only during prefilling;
(ii) preserving “sink” tokens; (iii) interleaving
FA/SWA layers; (iv) chain-of-thought (CoT);
and (v) fine-tuning. Our experiments show that
SWA adaptation is feasible while non-trivial:
no single method suffices, yet specific synergistic combinations effectively recover the original
long-context performance. We further analyze
the performance-efficiency trade-offs of different SWAA configurations and provide recommended recipes for diverse scenarios. Our code
[is available at github.](https://github.com/yuyijiong/sliding-window-attention-adaptation)


**1** **Introduction**


Transformer-based Large Language Models
(LLMs) (Vaswani et al., 2017) demonstrate
remarkable capabilities, but their self-attention
scales quadratically with the input sequence length,
making long context processing inefficient. Sliding
Window Attention (SWA), the most straightforward and widely adopted sparse attention
pattern, which restricts each token’s attention to a
fixed-size local window, reduces the computational
complexity to linearity, along with some other
benefits (see Appendix A).
To apply SWA to LLMs, typical solutions involve training a model with SWA from scratch, but



are prohibitively costly and cannot match the performance of state-of-the-art full-causal-attention

models like Qwen3 (Team, 2025b), mainly due
to the inability to reproduce pretraining data.
Training-free methods like streaming attention
(Xiao et al., 2024) can stabilize LLM outputs by retaining “sink tokens” while applying SWA, which
greatly improve efficiency but inevitably suffer
from severe long-context performance degradation
possibly due to the inaccessibility of distant tokens’
information (Xiao, 2025). This motivates a critical,
unexplored question: _Can a full-attention model_
_be adapted to sliding window attention at low cost_
_while maintaining long-context performance?_
We answer Yes to this question by proposing
Sliding Window Attention Adaptation(SWAA), a
set of recipes for adapting FA-pretrained models to
SWA, which requires neither costly pretraining nor
modifications to the standard Transformer architecture. Specifically, it systematically combines five
practical and composable methods:


1. **Full Attention (FA) Decode** : applying SWA
only during the prefilling stage while switching back to full attention for decoding.


2. **Keep First** _k_ **Tokens** : explicitly preserving
attention to the first _k_ “sink” tokens.


3. **Interleaving FA/SWA layers** : mix fullattention and SWA layers (e.g., assigning
SWA to half layers).


4. **Chain-of-Thought (CoT)** : enforcing an explicit "thinking" process during decoding.


5. **Fine-tuning with SWA** : lightweight SWAaware supervised fine-tuning on long-context
data.


Among these, FA Decode is a novel method we
introduce. Keep First _k_ Tokens and FA/SWA Interleaving have been proven effective in prior work



1


(Xiao et al., 2024; Team, 2024a; Zhang et al., 2024),
while CoT and fine-tuning are common LLM techniques. However, how these methods should be
combined to be actually effective for SWA adaptation remains unexplored.
Therefore, in our experiments, we evaluate
SWAA on Qwen3 (Team, 2025b) and Llama3.1
(Team, 2024b) across several long-context benchmarks, measuring both performance and efficiency
under a wide range of SWAA recipes. First, we
find that each method makes a distinct contribution,
but no single ingredient suffices to make SWA competitive with full attention. Second, we show that
specific synergistic combinations of methods can
recover a large fraction of the original long-context
performance. Third, we analyze the performanceefficiency trade-offs of different SWAA recipes and
identify some recommended configurations suitable for different deployment scenarios.
Rather than proposing a single globally optimal
configuration, we view SWAA as a flexible toolkit
of practical recipes: practitioners can select SWAA
recipes that match their accuracy and efficiency
constraints, or compose their own SWA adaptation
strategies by combining the available ingredients.
Our key contributions are:


1. We perform the first systematic study on adapting FA-pretrained LLMs to SWA without pretraining, revealing novel insights about how
SWA impacts LLMs and providing a foundation for future research in efficient sparse
attention.


2. We propose SWAA, a set of practical
SWA adaptation recipes that offer a robust
performance-efficiency balance for various
use cases, accelerating LLM inference from
the bottom level.


3. We implement our methods with FlashAttention (Dao, 2024) and vLLM (Kwon
et al., 2023), making it plug-and-play and userfriendly for practical deployment.


**2** **Related Works**


The _O_ ( _N_ [2] ) complexity of self-attention in Transformers (Vaswani et al., 2017) has spurred a wide
field of research about more efficient language
model architectures. Among the two most popular technological routes are sparse attention and
linear attention.



**2.1** **Sparse Attention**


Our work falls in this category. Sliding Window
Attention (SWA) represents the most basic form
of local sparse attention, yet its performance is inherently limited. Therefore, model architectures
such as Longformer (Beltagy et al., 2020), BigBird
(Zaheer et al., 2020), and RATTENTION (Wang
et al., 2025) combine local SWA on most tokens
with special global attention on specific tokens to
create a more powerful, albeit still sparse, pattern.
Popular LLMs like Gemma2 (Team, 2024a) adopt
SWA in half of their layers to balance the efficiency
of SWA and peformance of FA. Sliding Window
Attention Training (SWAT) (Fu et al., 2025b) introduces architectural changes, such as sigmoid
activation and balanced position embeddings, to
stabilize SWA performance. More advanced methods like Deepseek-sparse-attention (Yuan et al.,
2025; DeepSeek-AI, 2025b), although achieving
excellent quality, involve more complicated implementation and optimization due to semantic-aware
attention operations (e.g., selecting the most important tokens based on attention weights). Regardless,
almost all of the above methods require pretraining with a specific sparse pattern, which is costly
and fails to leverage the advantages of existing pretrained models.


LightTransfer (Zhang et al., 2024) is a promising
attempt at adapting existing models to SWA without pretraining, which has the same motivation as
ours. But it may generalize poorly across model
families (see Appendix G).


**2.2** **Linear Attention**


An alternative approach involves reformulating
the attention mechanism entirely to achieve linear, _O_ ( _N_ ), complexity. This includes methods
such as RNN-like linear attention transformers

(Katharopoulos et al., 2020; Peng et al., 2023;
Sun et al., 2023) and structured state-space models
(SSMs) like Mamba (Gu and Dao, 2023). Many
works such as Jamba and Nemotron-Flash(Lieber
et al., 2024; Linsong Chu et al., 2024; Team et al.,
2025; Fu et al., 2025a) interleave linear attention
layers with traditional attention layers to create hybrid model structures. While promising, they represent a fundamental architectural departure from the
standard Transformer, and their performance is generally weaker than traditional Transformer-based

LLMs.



2


**3** **Candidate Methods for SWA**

**Adaptation**


As established, a naive application of SWA leads
to severe long-context performance degradation.
Therefore, we investigate five methods that can potentially facilitate SWA adaptation from distinct
perspectives. However, every method except finetuning has some drawbacks to LLM inference efficiency, as discussed in Appendix A. Therefore,
although these methods are not mutually exclusive,
we should not indiscriminately adopt all of them.
Instead, we must evaluate various combinations to
identify the optimal trade-off between performance
and efficiency.


**3.1** **Full Attention Decode**


This method applies SWA **only** to the prefilling
stage. During the decoding (auto-regressive generation) stage, each token still employs full attention,
allowing access to all previous tokens in the context. The resulting attention mask is depicted in
Figure 1a.
This novel approach, first proposed in our study,
is inspired by human reading comprehension: humans typically scan a passage casually (prefilling)
before thinking deeply to formulate an answer (decoding) for a specific problem. We term this strategy "reading casually, thinking carefully." In our
design, the SWA-constrained prefilling stage corresponds to casual reading, while the full-attention
decoding stage enables careful thinking. This analogy also suggests that Chain-of-Thought (CoT)
during decoding may be particularly beneficial, as
extended generation (i.e. "thinking") could compensate for the insufficient contextual information
gathered during the prefilling stage.


(a) FA Decode (b) Keep First


Figure 1: (a) Attention mask for FA Decode. SWA is
used for prompt tokens (prefill), and full attention is
used for generated tokens (decode). (b) Attention mask
for SWA combined with Keep First _k_ Tokens.



**3.1.1** **Keep First k Tokens**

Xiao et al. demonstrate that models pretrained with
full attention allocate a disproportionate amount
of attention to the initial tokens ("attention sink"),
and removing the visibility of these tokens causes
performance collapse. Their solution, streaming
attention, involves permanently retaining the attention to these "sink" tokens while using SWA, which
successfully maintains the stability of the attention
distribution and the model’s output. Our method
is basically the same: as shown in Figure 1b, any
subsequent token can attend to its local window
**and** the initial _k_ tokens.

Notably, however, our method extends beyond
its original version. Streaming attention operates
only at the KV cache level; specifically, the KV
cache of the sink tokens is retained during decoding, while the prefilling stage is not modified or accelerated at all. In contrast, we directly customize
the Flash-Attention-2 (Dao, 2024) kernel to implement such attention mask, thereby accelerating
prefilling via SWA as well, and eliminating the
need to modify KV cache.


**3.2** **Interleaving Layers**


This method retains full attention on a subset of

layers while applying SWA to the remainder, providing a simple hybrid mechanism to balance the
performance of full attention with the efficiency
of pure SWA. A common strategy involves designating one in every _n_ layers to use full attention
(e.g., layers 0, 2, 4, . . . retain full attention, while all
others use SWA). For example, Gemma-2 (Team,
2024a) uses SWA only for layers [1, 3, 5, . . . ], and
Gemma-3 (Team, 2025a) uses SWA only for layers

[5, 11, 17, ...].
However, for an FA-pretrained model, layers
may exhibit distinct behaviors, suggesting it may
not be optimal to simply assign SWA to layers [1,
3, 5, ...]. Instead, it might be preferable to use
statistical features to select "lazy" (mostly focusing
on just recent tokens) layers, as adopted by LightTransfer (Zhang et al., 2024). However, we find
that LightTransfer is not consistently superior in
practice (see Appendix G). Therefore, for simplicity, we still choose the simplest fixed-interval layer
selections in our experiments, such as [0, 2, 4, . . . ]
and [1, 3, 5, ...].


**3.3** **Chain-of-Thought**


Chain-of-Thought (CoT) (Wei et al., 2022) is a
widely used technique for improving model ac


3




```json
"img_sliding_window_attention_2_0": {
  "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/sliding_window_attention.pdf-2-0.png",
  "page": 2,
  "section": "3.3 Chain-of-Thought",
}
```


```json
"img_sliding_window_attention_2_1": {
  "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/sliding_window_attention.pdf-2-1.png",
  "page": 2,
  "section": "3.3 Chain-of-Thought",
}
```
curacy via reasoning. With the advent of "thinking" models, such as DeepSeek-R1 (DeepSeek-AI,
2025a), CoT has evolved from a prompting strategy
to an intrinsic LLM capability. However, whether
CoT has a specific impact in SWA scenarios remains uninvestigated. We explore this by comparing a thinking model with its non-thinking variant,
e.g., Qwen3-4B-Thinking and Qwen3-4B-Instruct,
to verify the effect of CoT on SWA adaptation,
which can produce more notable differences compared to simply adding CoT prompting to the same
model.


**3.4** **Fine-tuning**


This is the most natural way to mitigate traininginference mismatch. Apparently, the model should
be fine-tuned while SWA is applied, so that the
model’s parameters can be trained to better adapt
to SWA. Meanwhile, the training data must be longcontext tasks, where SWA actually works.
However, most available long-context datasets
only contain brief ground-truth answers, lacking the reasoning steps required for fine-tuning
a "thinking" model. Since our goal is to _restore_
the model’s original capabilities under SWA rather
than teach it new ones, instead of directly using the
original dataset, we adopt an approach similar to
self-distillation (Yang et al., 2024). Specifically, we
utilize the original full-attention model to generate
new answers for the dataset’s questions, and these
generated answers are then filtered for correctness
using GPT-5-Mini (OpenAI, 2025), to make up our
training dataset. For each question, we sample 4
answers with temperature 1, because we find this
strategy is slightly better than generating only one
answer with temperature 0.


**4** **Experiment Setup**


We organize our experiments around three research
questions:


**RQ1: Is SWA adaptation feasible without any**
**additional training?** We evaluate whether
an FA LLM can be adapted to SWA using
only inference-time modifications, and which
combinations of techniques are necessary.


**RQ2: How much does fine-tuning with SWA im-**
**prove performance?** We study the effect of
SWA-aware fine-tuning on long-context performance and identify which components of
SWAA are still required.



**RQ3: Which SWAA configurations achieves the**
**optimal performance-efficiency trade-offs?**
We evaluate how different SWAA configurations trade off accuracy against inference la
tency.


**4.1** **Models**


Our primary experiments use Qwen3-4B-Thinking
and Qwen3-4B-Instruct (Team, 2025b). The Thinking variant enforces chain-of-thought (CoT) style
reasoning, whereas the Instruct variant usually just
answers briefly. To ensure generality, we additionally evaluate Qwen3-30B-A3B-Thinking, Qwen330B-A3B-Instruct (Team, 2025b), and Llama3.18B-Instruct (Touvron et al., 2023).

All models are served with vLLM in float16

precision using a batch size of 64. We use greedy
decoding (temperature = 0) for all evaluations. In
preliminary experiments, we observed that vLLM
yields slightly lower (about 1% to 5%) scores
than HuggingFace Transformers due to precisionrelated discrepancies.


**4.2** **Evaluation Dataset**


SWA is identical to full attention when the context

length is within the window size. Even if the model
is fine-tuned, we can pre-calculate the prompt
length and simply disable the LoRA adapters for
short prompts to get completely the same response
as the original model. Therefore, our experiments
focus exclusively on long-context benchmarks with
inputs exceeding 16k tokens, as re-evaluating models on standard short-context benchmarks (e.g.,
MMLU (Hendrycks et al., 2021), GPQA (Rein
et al., 2023)) is completely unnecessary.
Since we find other long-context benchmarks are
either too easy or too difficult for 4B-level models (see Appendix B), we ultimately select LongMemEval (Wu et al., 2024), a benchmark consisting of various types of long-context QA tasks with
moderate difficulty, although it is originally designed for agent memory system evaluation. Its
context length is controllable by selecting a specific number of chat sessions to concatenate as the
context from a pool of hundreds of sessions (a session contains the chat history between user and
assistant within a day). To create a moderately difficult and discriminative evaluation, we construct
**LongMemEval_24k** by sampling 10 sessions, resulting in 500 samples ranging from 16k to 32k
with an average context length of 24k.



4


For additional validation of generalizability, as
shown in Appendix D, we also experiment on
LongBench-V2 (Bai et al., 2024b), a more modern and challenging benchmark that requires deep
reasoning across various real-world tasks.


**4.3** **Training Details**


For the fine-tuning dataset, we initially considered LongAlign (Bai et al., 2024a), a widely used
long-context fine-tuning dataset for adapt a regularlength model to long-context tasks. However, since

_∼_
its sample count ( 10,000) is insufficient, we incorporate an additional 6,000 samples from Fusangv1-long (Pan, 2024), a more comprehensive corpus
of over 40,000 long-context samples that includes
LongAlign as a subset.
We perform SWA-aware fine-tuning using LoRA
(Hu et al., 2022). Unless otherwise noted, we use
rank _r_ = 16 and _α_ = 128, and apply LoRA only
to the query, key, and value projection modules.
We adopt this parameter-efficient setting because
full-parameter fine-tuning often leads to overfitting
and degradation of the model’s original capabilities
in our preliminary experiments. We use a learning
rate of 1e-4 with a cosine decay schedule. Models
are fine-tuned for a single epoch on the sampled
long-context dataset since we observe no meaningful gains from additional epochs (see Appendix F).
Once training takes approximately 12 hours on an
8*H20 GPU server for Qwen3-4B and 30 hours for
Qwen3-30B-A3B.


**5** **Experiment Results**


**5.1** **SWA Adaptation Without Fine-tuning**


We first study SWA adaptation without any additional training. Table 1 reports LongMemEval_24k
accuracy for Qwen3-4B-Thinking ("think") and
Qwen3-4B-Instruct ("non-think") under different
combinations of SWAA components. In most settings, we use an aggressive 2k window to amplify
the impact of SWA. The configurations are ranked
by the number of methods applied (0, 1, 2, or 3 of
Interleaving Layers, Keep First and FA Decode).
Rows 1 (original model) and 2 (naive SWA) serve
as upper and lower baselines, respectively. In the
column "FA layers", the value records which layers use full attention, and [] means all the layers
use SWA, i.e., this method is not enabled. In the
column "keep first", the value is _k_ in Keep First
_k_ Tokens. When comparing results, an accuracy
difference of less than 5% is usually considered



statistically insignificant. From the results, we find
that:


**Naive SWA is not viable.** Naively replacing FA
with a 2k sliding window attention (row 1) drops
accuracy significantly to 3 _._ 2 and 11 _._ 0, respectively.
Even with an 8k window (row 2), accuracy only recovers to 13 _._ 2 and 19 _._ 8, far below the FA baseline.


**Single method helps, but cannot close the gap.**
Each method—Keep First, FA Decode, or Interleaving Layers—improves over naive SWA (rows 3–6),
yet each alone recovers only a small fraction of the
FA gap and remains well below the baseline. In
short, no single method is sufficient.


**Combinations exhibit strong synergy.** Recipes
that combine multiple methods deliver large gains:


  - **FA Decode + Keep First** _k_ substantially improves over naive SWA (rows 7–9), recovering roughly half to two-thirds of the gap on
the thinking model as _k_ increases. However,
increasing _k_ from 100 to 1000 yields almost
no improvement, indicating that _k_ does not
need to be exceedingly large.


  - **Interleaving Layers + FA Decode** is
markedly stronger (row 13), recovering most
of the gap for the thinking model.


  - **FA Decode + Interleaving Layers + Keep**
**First** _k_ perform best (rows 18). The thinking
model recovers close to 90% of the FA gap
even at 2k window.


**CoT synergizes with FA Decode.** Under recipes
that include FA Decode, the thinking model consistently benefits more than the non-thinking model
(rows 13 and 18), suggesting that CoT synergizes
with FA Decode: preserving global attention at
decoding time enables longer reasoning traces to
capitalize on context processed by SWA, confirming our hypothesis in Section 3.1.


**Sliding window size affects, but is not the deci-**
**sive role.** With FA Decode + Keep First _k_, accuracy improves as the window increases (rows 7,
14, 15), though benefits are modest until 8k. When
added with interleaving FA layers, moving from a
2k to a 4k window is enough to close the remaining
gap on the thinking model (row 22 matches the FA
baseline), indicating that FA Decode and Interleaving are the dominant drivers, and larger windows
mainly smooth residual error.



5


Table 1: Qwen3-4B-Thinking and Qwen3-4B-Instruct results on LongMemEval without SFT


**No.** **window size** **FA layers** **keep first** **FA decode** **Acc think** **Acc non-think**


0 Full [] 0 False **73.0** **62.0**
1 2k [] 0 False 3.2 11.0
2 8k [] 0 False 13.2 19.8


3 2k [] 10 False 16.0 15.6
4 2k [] 0 True 11.8 14.2
5 2k [1, 3, 5, ...] 0 False 13.4 18.4
6 8k [] 0 True 26.2 25.0


7 2k [] 10 True 38.2 20.6
8 2k [] 100 True 50.0 17.8
9 2k [] 1000 True 50.0 20.2
10 2k [0, 2, 4, ...] 10 False 17.0 14.8
11 2k [0, 2, 4, ...] 0 True 32.2 26.0
12 2k [1, 3, 5, ...] 10 False 25.8 36.4
13 2k [1, 3, 5, ...] 0 True 59.2 34.8
14 4k [] 10 True 38.0 24.4
15 8k [] 10 True 49.2 35.2


16 2k [0, 2, 4, ...] 10 True 36.0 17.2
17 2k [1, 3, 5, ...] 10 True 65.0 **53.6**
18 2k [1, 3, 5, ...] 100 True **68.8** 50.6
19 2k [1, 5, 9, ...] 10 True 53.2 31.4
20 2k [1, 9, 17, ...] 10 True 36.4 18.8
21 2k [3, 7, 11, ...] 10 True 54.2 34.6
22 4k [1, 3, 5, ...] 100 True **73.0** **54.2**
23 8k [1, 3, 5, ...] 100 True **71.6** **56.6**



**SWA/FA layer selection has large impacts.** Selecting less layers for FA, e.g. only one fourth or
eighth (row 19, 20), though more efficient, will
significantly decrease the improvement brought by
Interleaving Layers. More importantly, for Qwen34B, configuring _odd-numbered_ layers with full attention is significantly better than _even-numbered_
layers (row 11, 13, 16, 17). However, surprisingly,
this result is reversed for Qwen3-30B-A3B and
Llama3.1-8B-Instruct (row 10 and 11 in Table 4
and 5). This suggests that layer functionalities differ across model families and sizes, necessitating
model-specific layer selection strategies, as discussed in Section 3.2.


Therefore, we answer RQ1: adapting an FA
LLM to SWA is feasible even without any training. But it requires specific combinations of at
least 2 methods, which could be less efficient for
inference.



**5.2** **SWA Adaptation With Fine-tuning**


We next evaluate SWA-aware supervised finetuning, which is expected to provide higher improvement. Table 2 reports LongMemEval_24k
accuracy after SFT under various SWAA configurations. The original full-attention model is also
fine-tuned as a baseline (row 0). Since training is
relatively time-consuming, we only test a representative subset of configurations. Our findings are as
follows:


**Fine-tuning substantially lifts all SWA config-**
**urations.** Comparing all the fine-tuning results
with non-training ones, it is clear that fine-tuning
consistently provides great improvement. However,
simply fine-tuning with naive SWA remains insufficient, only achieving 18.8% and 23.8% accuracy
(row 1).


**FA Decode and Interleaving Layers emerge as**
**dominant components.** After SFT, FA Decode
and Interleaving Layers provide the largest gains.



6


Table 2: Qwen3-4B-Thinking and Qwen3-4B-Instruct results on LongMemEval with SFT


**No.** **window size** **FA layers** **keep first** **FA decode** **Acc think** **Acc non-think**


0 Full [] 0 True **74.6** **63.4**
1 2k [] 0 False 18.8 23.8


2 2k [] 10 False 15.6 /
3 2k [] 0 True 57.9 42.0
4 2k [1, 3, 5, ...] 0 False 63.6 54.6
5 4k [] 0 True 62.6 /


6 2k [] 10 True 56.7 /
7 2k [] 100 True 62.2 42.6
8 2k [1, 3, 5, ...] 0 True **73.2** **58.8**
9 2k [0, 2, 4, ...] 0 True **66.0** /
10 2k [1, 5, 9, ...] 0 True **68.8** 47.0


11 2k [1, 3, 5, ...] 100 True **73.2** **61.4**


Table 3: Recommended SWA adaptation recipes for different needs and scenarios. � means optional.


**Training** ? **Thinking Model?** **Prefer?** **FA Decode** **Interleaving Layers** **Keep First**


No No Any � � �
No Yes Efficiency � � �
No Yes Accuracy � � �


Yes Any Efficiency � � �
Yes Any Accuracy � � �



Using FA Decode alone (row 3) or using Interleaving Layers alone (row 4) both get high accuracy.
And combining both (row 8) further boosts performance to 73 _._ 2 (think) and 58 _._ 8 (non-think), nearly
matching the full-attention SFT baseline in row 0.


**Keep First becomes optional rather than essen-**
**tial.** Before fine-tuning, Keep First is crucial for
stability under SWA. But after SFT, it only provides
minor additional improvement. With FA Decode,
adding _k_ = 100 (row 7) improves over _k_ = 0
(row 3) only 4.5%, and if further combining FA
layers, it almost offers no improvements (row 11
and row 8).


**Effect of sliding window size.** Row 3 and 5
shows that increasing the window from 2k to 4k
with FA Decode improves thinking-model accuracy from 57 _._ 9 to 62 _._ 6. This mirrors the non-SFT
trend that larger windows help, but the dominant
improvements still come from FA Decode and Interleaving Layers.

So, we answer RQ2: fine-tuning brings remarkably high performance restoration, provided we ap


ply **FA Decode**, **Interleaving Layers**, or a combination thereof, while **Keep First** becomes optional.
And the improvement brought by SFT under each
configuration varies significantly, meaning a nearoptimal training-free configuration need not remain
optimal after SFT, and vice versa.


**5.3** **Performance–efficiency Trade-offs and**
**Recommended Recipes**


Although integrating more methods can typically
achieve higher accuracy, it introduces more overhead, indicating that the efficiency of each recipe
must also be evaluated. To assess the performanceefficiency trade-off of different SWAA configurations, we evaluate time-to-first-token (TTFT), timeper-output-token (TPOT), total throughput, and average running time per request. Concretely, we
benchmark Qwen3-4B-Thinking on a single H100
GPU using vLLM’s bench_serve utility (Kwon
et al., 2023) with random input data and 100 total
requests. The prompt length and output length are
set to 128k and 512 tokens, respectively, representing a typical long-context QA setting.



7



```json
"img_sliding_window_attention_7_0": {
  "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/sliding_window_attention.pdf-7-0.png",
  "page": 7,
  "section": "5.3 Performance–efficiency Trade-offs and",
}
```


```json
"img_sliding_window_attention_7_1": {
  "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/sliding_window_attention.pdf-7-1.png",
  "page": 7,
  "section": "5.3 Performance–efficiency Trade-offs and",
}
```

(a) Qwen3-4B-Thinking (b) Qwen3-4B-Instruct


Figure 2: Accuracy and inference time of each configuration of Qwen3-4B on LongMemEval



To visualize the performance-efficiency tradeoff, Figure 2 plots each configuration’s accuracy
on LongMemEval_24k (Wu et al., 2024) against its
average running time, while detailed TTFT, TPOT,
and throughput statistics for each configuration are
provided in Appendix E. We draw a line between
the full-attention point and the naive-SWA point
as a baseline curve: configurations above this line
offer a better accuracy-latency balance intuitively.
For configurations with nearly identical time costs,
we display only the one with the highest accuracy.
Since **Keep First** _k_ has negligible impact on runtime (Appendix E), all plotted configurations fix
_k_ = 10.


From Figure 2, we observe that many configurations in Figure 2 achieve a clearly better
performance-efficiency ratio than baselines. And
for the thinking model, more points lie above the
baseline curve compared to non-thinking, indicating that **CoT** generally has a positive effect on improving the performance-efficiency ratio of SWAA.


Thus, we finally answer RQ3: many SWAA
configurations all reach excellent performanceefficiency trade-off, but there is no single metric to
quantify such trade-off to decide the globally optimal one. We therefore summarize **recommended**

**SWA adaptation recipes** tailored to various deployment scenarios in Table 3. And we must note
that specific parameters should be flexibly set to
meet application-specific requirements, without the
need to follow our experimental parameters (e.g.,



a 2k window, _k_ = 10). For example, you can increase the window size to 4k or _k_ to 128 for higher
accuracy and acceptable additional overhead.


**6** **Conclusion**


In this work, we validate the feasibility of adapting
full-attention pretrained LLMs to Sliding Window
Attention (SWA) for better efficiency, offering a
cost-effective alternative that avoids training sparseattention models from scratch. By systematically
deconstructing the adaptation process, we identify that the catastrophic degradation observed in
naive implementations can be effectively mitigated
through synergistic combinations of auxiliary methods. Our extensive experiments across the Qwen
and Llama families demonstrate that while trade
offs between computational overhead and model
performance are inevitable, optimized configurations can get an excellent performance-efficiency
balance.


**7** **Limitations**


We speculate that the ideal reasoning trajectory of
the model adapted to SWA should be longer than
the original model, to compensate for the information loss caused by SWA. That means, using
the answers generated by the original model as
fine-tuning data may not be the optimal training
method. Rather, RL methods like GRPO (Shao
et al., 2024) might further help the model adapted
to SWA learn a better reasoning trajectory. How


8


ever, we did not experiment with them since they
are too time-consuming and unstable.
We have not yet implemented the KV cache eviction (or overwriting) mechanism when using SWA;
that is, although the speed is improved, memory
usage is not effectively reduced.
Further experiments may be needed to confirm
whether our conclusions generalize to larger model
sizes, such as 70B.


**References**


Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei
Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. 2024a.
[Longalign: A recipe for long context alignment of](https://arxiv.org/abs/2401.18058)
[large language models.](https://arxiv.org/abs/2401.18058)


Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
[and Juanzi Li. 2023. Longbench: A bilingual, multi-](https://arxiv.org/abs/2308.14508)
[task benchmark for long context understanding.](https://arxiv.org/abs/2308.14508)


Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024b.
[Longbench v2: Towards deeper understanding and](https://arxiv.org/abs/2412.15204)
[reasoning on realistic long-context multitasks.](https://arxiv.org/abs/2412.15204)


Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020.

[Longformer: The long-document transformer.](https://arxiv.org/abs/2004.05150)


[Tri Dao. 2024. Flashattention-2: Faster attention with](https://openreview.net/forum?id=mZn2Xyh9Ec)
[better parallelism and work partitioning.](https://openreview.net/forum?id=mZn2Xyh9Ec) In _The_
_Twelfth International Conference on Learning Rep-_
_resentations, ICLR 2024, Vienna, Austria, May 7-11,_
_2024_ . OpenReview.net.


[DeepSeek-AI. 2025a. Deepseek-r1: Incentivizing rea-](https://arxiv.org/abs/2501.12948)
[soning capability in llms via reinforcement learning.](https://arxiv.org/abs/2501.12948)


[DeepSeek-AI. 2025b. Deepseek-v3.2: Pushing the fron-](https://arxiv.org/abs/2512.02556)
[tier of open large language models.](https://arxiv.org/abs/2512.02556)


Yonggan Fu, Xin Dong, Shizhe Diao, Hanrong Ye, Wonmin Byeon, Yashaswi Karnati, Lucas Liebenwein,
Maksim Khadkevich, Alexander Keller, Jan Kautz,
et al. 2025a. Nemotron-flash: Towards latencyoptimal hybrid small language models. In _The Thirty-_
_ninth Annual Conference on Neural Information Pro-_
_cessing Systems_ .


Zichuan Fu, Wentao Song, Yejing Wang, Xian Wu,
Yefeng Zheng, Yingying Zhang, Derong Xu, Xuetao
[Wei, Tong Xu, and Xiangyu Zhao. 2025b. Sliding](https://arxiv.org/abs/2502.18845)
[window attention training for efficient large language](https://arxiv.org/abs/2502.18845)
[models.](https://arxiv.org/abs/2502.18845)


[Albert Gu and Tri Dao. 2023. Mamba: Linear-time](https://arxiv.org/abs/2312.00752)
[sequence modeling with selective state spaces.](https://arxiv.org/abs/2312.00752)



Dan Hendrycks, Collin Burns, Steven Basart, Andy
Zou, Mantas Mazeika, Dawn Song, and Jacob Stein[hardt. 2021. Measuring massive multitask language](https://openreview.net/forum?id=d7KBjmI3GmQ)
[understanding. In](https://openreview.net/forum?id=d7KBjmI3GmQ) _9th International Conference on_
_Learning Representations, ICLR 2021, Virtual Event,_
_Austria, May 3-7, 2021_ . OpenReview.net.


Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang,
[and Boris Ginsburg. 2024. Ruler: What’s the real](https://arxiv.org/abs/2404.06654)
[context size of your long-context language models?](https://arxiv.org/abs/2404.06654)


Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
[Weizhu Chen. 2022. Lora: Low-rank adaptation of](https://openreview.net/forum?id=nZeVKeeFYf9)
[large language models. In](https://openreview.net/forum?id=nZeVKeeFYf9) _The Tenth International_
_Conference on Learning Representations, ICLR 2022,_
_Virtual Event, April 25-29, 2022_ . OpenReview.net.


Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pap[pas, and François Fleuret. 2020. Transformers are](http://proceedings.mlr.press/v119/katharopoulos20a.html)
[rnns: Fast autoregressive transformers with linear](http://proceedings.mlr.press/v119/katharopoulos20a.html)
[attention. In](http://proceedings.mlr.press/v119/katharopoulos20a.html) _Proceedings of the 37th International_
_Conference on Machine Learning, ICML 2020, 13-18_
_July 2020, Virtual Event_, volume 119 of _Proceedings_
_of Machine Learning Research_, pages 5156–5165.
PMLR.


Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
[Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-](https://arxiv.org/abs/2309.06180)
[cient memory management for large language model](https://arxiv.org/abs/2309.06180)
[serving with pagedattention.](https://arxiv.org/abs/2309.06180)


Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen,
Jhonathan Osin, Itay Dalmedigos, Erez Safahi,
Shaked Meirom, Yonatan Belinkov, Shai ShalevShwartz, Omri Abend, Raz Alon, Tomer Asida,
Amir Bergman, Roman Glozman, Michael Gokhman,
Avashalom Manevich, Nir Ratner, Noam Rozen,
Erez Shwartz, Mor Zusman, and Yoav Shoham.
[2024. Jamba: A hybrid transformer-mamba language](https://arxiv.org/abs/2403.19887)
[model.](https://arxiv.org/abs/2403.19887)


Tri Dao Linsong Chu, Divya Kumari et al. 2024.

[Bamba: Inference-efficient hybrid mamba2 model.](https://huggingface.co/blog/bamba)


[OpenAI. 2025. ChatGPT.](https://chatgpt.com/)


Wenbo Pan. 2024. [Fusang-v1: A large curation of](https://huggingface.co/datasets/wenbopan/Fusang-v1)
[instruction-tuning datasets for better bilingual and](https://huggingface.co/datasets/wenbopan/Fusang-v1)
[long-range llms.](https://huggingface.co/datasets/wenbopan/Fusang-v1)


Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi
Cao, Xin Cheng, Michael Chung, Leon Derczynski,
Xingjian Du, Matteo Grella, Kranthi Gv, Xuzheng
He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej Koptyra, Hayden
Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand
Mom, Atsushi Saito, Guangyu Song, Xiangru Tang,
Johan Wind, Stanisław Wo´zniak, Zhenyuan Zhang,
Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. 2023.
[RWKV: Reinventing RNNs for the transformer era.](https://doi.org/10.18653/v1/2023.findings-emnlp.936)
In _Findings of the Association for Computational_
_Linguistics: EMNLP 2023_, pages 14048–14077, Singapore. Association for Computational Linguistics.



9


David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Ju[lian Michael, and Samuel R. Bowman. 2023. Gpqa:](https://arxiv.org/abs/2311.12022)
[A graduate-level google-proof q&a benchmark.](https://arxiv.org/abs/2311.12022)


Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024.
[Deepseekmath: Pushing the limits of mathematical](https://arxiv.org/abs/2402.03300)
[reasoning in open language models.](https://arxiv.org/abs/2402.03300)


Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma,
Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu
[Wei. 2023. Retentive network: A successor to trans-](https://arxiv.org/abs/2307.08621)
[former for large language models.](https://arxiv.org/abs/2307.08621)


[Gemma Team. 2024a. Gemma 2: Improving open lan-](https://arxiv.org/abs/2408.00118)
[guage models at a practical size.](https://arxiv.org/abs/2408.00118)


[Gemma Team. 2025a. Gemma 3 technical report.](https://arxiv.org/abs/2503.19786)


Ling Team, Bin Han, Caizhi Tang, Chen Liang, Donghao Zhang, Fan Yuan, Feng Zhu, Jie Gao, Jingyu Hu,
Longfei Li, Meng Li, Mingyang Zhang, Peijie Jiang,
Peng Jiao, Qian Zhao, Qingyuan Yang, Wenbo Shen,
Xinxing Yang, Yalin Zhang, Yankun Ren, Yao Zhao,
Yibo Cao, Yixuan Sun, Yue Zhang, Yuchen Fang,
[Zibin Lin, Zixuan Cheng, and Jun Zhou. 2025. Every](https://arxiv.org/abs/2510.19338)
[attention matters: An efficient hybrid architecture for](https://arxiv.org/abs/2510.19338)
[long-context reasoning.](https://arxiv.org/abs/2510.19338)


[Llama Team. 2024b. The llama 3 herd of models.](https://arxiv.org/abs/2407.21783)


[Qwen3 Team. 2025b. Qwen3 technical report.](https://arxiv.org/abs/2505.09388)


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, Aurelien Rodriguez, Armand Joulin, Edouard
[Grave, and Guillaume Lample. 2023. Llama: Open](https://arxiv.org/abs/2302.13971)
[and efficient foundation language models.](https://arxiv.org/abs/2302.13971)


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
[Kaiser, and Illia Polosukhin. 2017. Attention is all](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
[you need. In](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) _Advances in Neural Information Pro-_
_cessing Systems 30: Annual Conference on Neural_
_Information Processing Systems 2017, December 4-9,_
_2017, Long Beach, CA, USA_, pages 5998–6008.


Bailin Wang, Chang Lan, Chong Wang, and Ruoming
[Pang. 2025. Rattention: Towards the minimal sliding](https://arxiv.org/abs/2506.15545)
[window size in local-global attention models.](https://arxiv.org/abs/2506.15545)


Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le,
[and Denny Zhou. 2022. Chain-of-thought prompting](http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
[elicits reasoning in large language models. In](http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) _Ad-_
_vances in Neural Information Processing Systems 35:_
_Annual Conference on Neural Information Process-_
_ing Systems 2022, NeurIPS 2022, New Orleans, LA,_
_USA, November 28 - December 9, 2022_ .


Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang,
[Kai-Wei Chang, and Dong Yu. 2024. Longmemeval:](https://arxiv.org/abs/2410.10813)
[Benchmarking chat assistants on long-term interac-](https://arxiv.org/abs/2410.10813)
[tive memory.](https://arxiv.org/abs/2410.10813)



Guangxuan Xiao. 2025. Why stacking sliding win[dows can’t see very far. https://guangxuanx.com/](https://guangxuanx.com/blog/stacking-swa.html)
[blog/stacking-swa.html.](https://guangxuanx.com/blog/stacking-swa.html)


Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
[Han, and Mike Lewis. 2024. Efficient streaming lan-](https://openreview.net/forum?id=NG7sS51zVF)
[guage models with attention sinks. In](https://openreview.net/forum?id=NG7sS51zVF) _The Twelfth_
_International Conference on Learning Representa-_
_tions, ICLR 2024, Vienna, Austria, May 7-11, 2024_ .
OpenReview.net.


Zhaorui Yang, Tianyu Pang, Haozhe Feng, Han Wang,
[Wei Chen, Minfeng Zhu, and Qian Liu. 2024. Self-](https://arxiv.org/abs/2402.13669)
[distillation bridges distribution gap in language](https://arxiv.org/abs/2402.13669)
[model fine-tuning.](https://arxiv.org/abs/2402.13669)


Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo,
Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing
Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong
Ruan, Ming Zhang, Wenfeng Liang, and Wangding
Zeng. 2025. [Native sparse attention: Hardware-](https://doi.org/10.18653/v1/2025.acl-long.1126)
[aligned and natively trainable sparse attention. In](https://doi.org/10.18653/v1/2025.acl-long.1126)
_Proceedings of the 63rd Annual Meeting of the As-_
_sociation for Computational Linguistics (Volume 1:_
_Long Papers)_, pages 23078–23097, Vienna, Austria.
Association for Computational Linguistics.


Manzil Zaheer, Guru Guruganesh, Kumar Avinava
Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang,
[Li Yang, and Amr Ahmed. 2020. Big bird: Trans-](https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html)
[formers for longer sequences. In](https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html) _Advances in Neural_
_Information Processing Systems 33: Annual Confer-_
_ence on Neural Information Processing Systems 2020,_
_NeurIPS 2020, December 6-12, 2020, virtual_ .


Xuan Zhang, Fengzhuo Zhang, Cunxiao Du, Chao Du,
[Tianyu Pang, Wei Gao, and Min Lin. 2024. Light-](https://arxiv.org/abs/2410.13846)
[transfer: Your long-context llm is secretly a hybrid](https://arxiv.org/abs/2410.13846)
[model with effortless adaptation.](https://arxiv.org/abs/2410.13846)


**A** **SWA’s Benefits and Each Method’s**
**Drawbacks**


SWA reduces the computational complexity to
_O_ ( _N · W_ ), where _W_ is the window size. The
benefits are threefold: (1) SWA reduces the computational load, (2) conserves GPU memory by limiting the required Key-Value (KV) cache, and (3)
enhances KV cache reusability beyond traditional
prefix caching, since a token’s state is independent
of tokens outside its local window. However, there

is no free lunch—each method of SWAA has some

drawbacks, impairing the benefits brought by SWA
to varying degrees.
**FA Decode** presents two primary drawbacks:
(1) the benefits apply only to prefilling, while decoding speed is not accelerated as it utilizes full
attention, and (2) the GPU memory required for
the KV cache is not reduced, as the KV cache for
the full context must be retained for decoding. In



10


practice, however, many distributed LLM services
have to recompute the KV cache of the entire chat
history because storing and loading the KV cache
complicates engineering systems, making prefilling occurs more frequently than expected, thereby
amplifying the advantage of this method.
**Keep First** introduces very minor computational
overhead, but it complicates efficient KV cache
reuse. Due to positional encoding, a token’s KV
state depends on its position relative to the initial
_k_ tokens, hindering simple cache reuse across different requests. A position encoding separation or
offsetting mechanism may be needed.
**Interleaving Layers** introduces the most significant overhead, as only a subset of layers benefits from the computational savings of SWA. Furthermore, the GPU memory required for the KV
cache is not reduced for the full-attention layers.
Additionally, this method negates the KV cache
reusability advantage of SWA, as the existence of
full-attention layers violates the independence of
the KV cache beyond the local window.
**CoT** will greatly increase the generation length,
especially for difficult tasks. So the decoding time
will be much longer.


**B** **Other Long-context Benchmarks**


We find existing long-context benchmarks problematic for our specific needs. For example:


1. LongBench (Bai et al., 2023) is classic and
widely used, but its average context length
(most are under 16k) is relatively short for
modern models, i.e., it is already too easy.
And its data source is too old, leading to a risk
of test data leakage.


2. Ruler(Hsieh et al., 2024) has controllable context length, but its tasks are almost all synthetic and most of them are needle-retrieval

tasks, failing to reflect the model’s overall
long-context capability in real-world scenarios.


3. LongBench-V2 (Bai et al., 2024b) is welldesigned to necessitate deep understanding
and reasoning over very long context. But it
is too challenging for 4B-level models (e.g.,
Qwen3-4B-Thinking only gets 35% accuracy,
which is too close to the random guessing
baseline of 25%), making the improvement of
different methods less distinguishable. Moreover, since it is in a multiple-choice question



format, the results may not be sufficiently reliable because the model has a 25% chance of

guessing the correct option.


However, despite the extreme difficulty of
LongBench-V2 (Bai et al., 2024b), it remains
a high-quality long-context benchmark after
all. Thus we still elect to conduct our experiments on it to verify the generalizability of
our conclusions, as shown in Appendix D.


**C** **Results of Other Models**


We show the results of Qwen3-30B-A3BThinking and Qwen3-30B-A3B-Instruct on LongMemEval_24k in Table 4, and the results of

Llama3.1-8B-Instruct in Table 5. The scores of

Qwen3-30B-A3B are generally higher and those
of Llama3.1 are generally lower, but all results are
consistent with our previous conclusions, demonstrating their generalizability.
Due to the time-intensive nature of training, we
only test a small set of configurations with finetuning.


**D** **Results of LongBench V2**


We present the results of LongBench V2 (Bai et al.,

2024b) in Tables 6, 7 and 8. We retain only the
samples whose context length is under 128k due to
GPU memory limitations; thus, 384 of 500 samples
are kept. However, due to the high difficulty, the
performance is generally poor. Some scores are
even below the random guessing baseline (25%).
For Qwen3-4B and Qwen3-30B-A3B models,

the results show less noticeable differences be
tween various methods. But fortunately, the trend
of accuracy changes is generally consistent with
that of other datasets, so they do not conflict with
all of our previous conclusions. For Llama3.1, due
to its weaker long-context capability, accuracy consistently hovers around 30%.


**E** **Inference Efficiency**


The TTFT, TPOT and total throughput when using vLLM are shown in Table 9. Since inference
speed is highly dependent on hardware, implementation details, and workload characteristics, these
numbers should be interpreted as reference values.
However, from the results, we can still conclude

that:


1. Interleaving Layers and FA Decode significantly slow down the speed compared to pure



11


SWA.


2. Keep First _k_ Tokens has a negligible impact
on efficiency.


3. Increasing the window size slightly increases
inference time. For example, increasing from
2k to 4k decreases throughput by only 10%,
but a 4k window generally achieves higher accuracy based on previous experiments. Therefore, in practice, a 4k window is a more common choice.


In theory, FA Decode should yield a decoding
speed identical to that of full attention. Yet, in this
table, we observe acceleration on TPOT. This is because vLLM-v1 typically mixes different requests’
prefilling and decoding tokens in one sequence
to improve GPU utilization. Thus, the speeds of
prefilling and decoding may affect each other. If
processing only a single request, the situation differs. For example, when the generation length is
set to 2000, we find decoding takes over 95% of the
total time, rendering the acceleration of the prefilling stage negligible—i.e., SWA with FA Decode is
almost unable to improve efficiency in such cases.


**F** **Influence of Training Epochs**


As shown in Table 10, training for more than
1 epoch yields no improvement. Therefore, we
choose to train for only 1 epoch.


**G** **Results of LightTransfer**


LightTransfer (Zhang et al., 2024) represents a
promising attempt at SWA adaptation on fullattention models without pretraining. It proposes
a layer selection method for SWA adaptation that
calculates a "lazy ratio," represented by the ratio
of attention from tokens at the end of the sequence
(from a calibration dataset) to recent tokens versus
global tokens. Layers with a higher "lazy ratio" are
selected to apply SWA, while the rest retain full
attention. This method is intuitive and theoretically
sound, but our experiments reveal some negative
results.

Since the complete code of LightTransfer is not
open-source, we reproduce this method using LongAlign (Bai et al., 2024a) as the calibration dataset
for lazy layer detection, where the number of last
tokens is set to 64, and the recent token window is
set to 1024. From our experimental results shown
in Table 11, we find that:



1. For Qwen3-4B, LightTransfer even has a
counterproductive effect; allowing lazy layers to use FA yields higher scores, while following the original method (letting non-lazy
layers use FA) results in significantly lower

scores.


2. For Qwen3-30B, it provides nearly no improvement over fixed-interval selection.


3. Only for Llama3.1-8B does LightTransfer
show advantages.


Therefore, we conclude that LightTransfer does
not yield stable performance across various models.
Although fine-grained layer selection methods are
theoretically superior, we believe they require further investigation before integration into our SWAA
recipes.



12


Table 4: Results of Qwen3-30B-A3B-Thinking and Qwen3-30B-A3B-Instruct on LongMemEval


**No.** **SFT** **window size** **FA layers** **keep first** **FA decode** **Acc think** **Acc non-think**


0 False Full [] 0 False **79.2** **71.6**
1 False 2k [] 0 False 0.0 0.4
2 False 8k [] 0 False 0.0 0.2


3 False 2k [] 10 False 0.0 2.8
4 False 2k [] 0 True 0.2 0.2
5 False 2k [0, 2, 4, ...] 0 False 21.0 28.4


6 False 2k [] 10 True 43.8 23.6
7 False 2k [] 100 True 58.6 22.2
8 False 2k [] 1000 True 59.0 25.4
9 False 4k [] 10 True 49.8 26.6


10 False 2k [0, 2, 4, ...] 10 True **74.8** **63.0**
11 False 2k [1, 3, 5, ...] 10 True 51.6 24.0
12 False 2k [0, 4, 8, ...] 10 True 48.8 23.8
13 False 2k [2, 6, 10, ...] 10 True 64.8 44.2
14 False 4k [0, 2, 4, ...] 100 True **74.6** **64.4**


15 True _\_ [] 0 False **79.6** _\_


16 True 2k [] 0 True 62.2 51.0


17 True 2k [] 100 True 65.6 50.8
18 True 2k [0, 2, 4, ...] 0 True **72.6** _\_


19 True 2k [0, 2, 4, ...] 100 True **77.8** **68.0**


Table 5: Results of Llama3.1-8B-Instruct on LongMemEval


**No.** **SFT** **window size** **FA layers** **keep first** **FA decode** **Acc non-think**


0 False Full [] 0 False **61.0**
1 False 2k [] 0 False 0.6
2 False 8k [] 0 False 1.2


3 False 2k [] 10 False 1.8
4 False 2k [] 0 True 0.0
5 False 2k [0, 2, 4, ...] 0 False 3.0


6 False 2k [] 10 True 16.8
7 False 2k [] 100 True 20.0
8 False 2k [] 1k True 24.2
9 False 4k [] 10 True 23.8


10 False 2k [0, 2, 4, ...] 10 True **42.6**
11 False 2k [1, 3, 5, ...] 10 True 21.0
12 False 2k [0, 4, 8, ...] 10 True 17.8
13 False 2k [2, 6, 10, ...] 10 True 24.4
14 False 4k [0, 2, 4, ...] 100 True **44.0**


13


Table 6: Qwen3-4B-Thinking and Qwen3-4B-Instruct results on LongBench-V2


**No.** **SFT** **window size** **FA layers** **keep first** **FA decode** **Acc think** **Acc non-think**


0 False Full [] 0 False **34.6** **35.2**
1 False 2k [] 0 False 9.4 25.8
2 False 8k [] 0 False 15.1 22.1


3 False 2k [] 10 False 7.7 25.8
4 False 2k [] 0 True **26.2** 25.2
5 False 2k [1, 3, 5, ...] 0 False 12.1 23.5
6 False 8k [] 0 True 22.8 25.5


7 False 2k [] 10 True 25.8 25.2
8 False 2k [] 100 True 24.2 26.5
9 False 2k [] 1000 True 23.8 25.2
10 False 2k [0, 2, 4, ...] 10 False 19.8 **29.2**
11 False 2k [0, 2, 4, ...] 0 True 23.8 **29.9**
12 False 2k [1, 3, 5, ...] 10 False 21.1 **29.9**
13 False 2k [1, 3, 5, ...] 0 True **28.5** 26.8
14 False 4k [] 10 True **27.9** 27.5


15 True Full [] 0 False **37.9** **34.9**
16 True 2k [] 0 False 7.4 30.9


17 True 2k [] 100 False 6.0 _\_
18 True 2k [] 0 True 29.2 30.2
19 True 2k [1, 3, 5, ...] 0 False 29.5 31.9
20 True 4k [] 0 True **32.9** _\_


21 True 2k [] 10 True 28.9 29.2
22 True 2k [] 100 True 29.2 30.5
23 True 2k [0, 2, 4, ...] 0 True 31.5 _\_
24 True 2k [0, 4, 8, ...] 0 True 30.9 _\_
25 True 2k [1, 3, 5, ...] 0 True **38.3** **34.6**
26 True 2k [1, 5, 9, ...] 0 True 32.0 32.2


27 True 2k [1, 3, 5, ...] 100 True **37.2** **33.9**


14


Table 7: Qwen3-30B-A3B-Thinking and Qwen3-30B-A3B-Instruct results on LongBench-V2


**No.** **SFT** **window size** **FA layers** **keep first** **FA decode** **Acc think** **Acc non-think**


0 False Full [] 0 False **49.7** **42.6**
1 False 2k [] 0 False 0.0 0.0
2 False 8k [] 0 False 0.0 0.0


3 False 2k [] 10 False 9.1 32.2
4 False 2k [] 0 True 0.0 0.0
5 False 2k [0, 2, 4, ...] 0 False 20.1 25.8


6 False 2k [] 10 True 9.1 32.2
7 False 2k [] 100 True 10.4 28.2
8 False 2k [] 1k True 11.7 29.5
9 False 4k [] 10 True 26.8 30.9


10 False 2k [0, 2, 4, ...] 10 True 22.1 **33.6**
11 False 2k [0, 4, 8, ...] 10 True 12.4 29.5
12 False 2k [1, 3, 5, ...] 10 True **30.2** 28.9
13 False 2k [2, 6, 10, ...] 10 True 21.1 35.6
14 False 4k [0, 2, 4, ...] 100 True **29.5** **35.9**


15 True Full [] 0 False **43.6** _\_


16 True 2k [] 0 True 35.9 33.9


17 True 2k [] 100 True **36.6** 32.9
18 True 2k [0, 2, 4, ...] 0 True **41.3** _\_


19 True 2k [0, 2, 4, ...] 100 True **48.0** **37.9**


Table 8: Llama3.1-8B-Instruct results on LongBench-V2


**No.** **SFT** **window size** **FA layers** **keep first** **FA decode** **Acc non-think**


0 False Full [] 0 False **33.2**
1 False 2k [] 0 False 0.0
2 False 8k [] 0 False 0.0


3 False 2k [] 10 False 28.9
4 False 2k [] 0 True 0.0
5 False 2k [0, 2, 4, ...] 0 False 0.0


6 False 2k [] 10 True 28.9
7 False 2k [] 100 True **30.2**
8 False 2k [] 1000 True **30.2**
9 False 4k [] 10 True 27.5


10 False 2k [0, 2, 4, ...] 10 True 28.2
11 False 2k [0, 4, 8, ...] 10 True **32.6**
12 False 2k [1, 3, 5, ...] 10 True 26.5
13 False 2k [2, 6, 10, ...] 10 True 26.8
14 False 4k [0, 2, 4, ...] 100 True **30.9**


15


Table 9: Efficiency metrics of different SWAA configurations on vLLM. "FA layers = 1/4" means one fourth of total
layers use full attention while the others use SWA.


**window** **keep first** **FA decode** **FA layers** **TTFT (s)** **TPOT (s)** **Throughput (k tks/s)**


Full 0 False None 1681.44 0.16 3.74


2k 0 False None 203.20 0.02 30.72

2k 100 False None 207.74 0.02 30.65

2k 0 False 1/2 938.00 0.09 6.70

2k 0 True None 963.39 0.11 6.39

2k 0 True 1/2 1321.39 0.14 4.72

2k 0 True 1/4 1141.66 0.12 5.43


4k 0 False None 233.07 0.02 27.03

4k 100 False None 237.87 0.02 26.74

4k 0 False 1/2 949.02 0.09 6.64

4k 0 True None 990.00 0.11 6.23

4k 0 True 1/2 1340.91 0.14 4.64

4k 0 True 1/4 1166.69 0.13 5.32


Table 10: Results of different training epochs of Qwen3-4B-Thinking on LongMemEval


**SFT (epochs)** **window size** **FA layers** **keep first** **FA decode** **Acc**


1 2k [] 0 True 58.0
2 2k [] 0 True 57.6
3 2k [] 0 True 56.0


Table 11: Results of LightTransfer on LongMemEval. "lazy" represents the half layers with higher lazy ratio, i.e.
those which should apply SWA in theory. "non-lazy" represents the other part, i.e. those which should keep full
attention.


**SFT** **window size** **FA layers** **keep first** **FA decode** **Acc think** **Acc non-think**


**Model Group: Qwen3-4B**


False 2k [0, 2, 4, ...] 100 True 48.8 18.4
False 2k [1, 3, 5, ...] 100 True **70.8** **50.4**
False 2k lazy 100 True 70.2 47.8
False 2k non-lazy 100 True 54.0 19.6


**Model Group: Qwen3-30B-A3B**


False 2k [0, 2, 4, ...] 100 True **75.8** **64.2**
False 2k [1, 3, 5, ...] 100 True 60.2 25.8
False 2k lazy 100 True 61.8 25.2
False 2k non-lazy 100 True 74.8 59.2


**Model Group: Llama3.1-8B-Instruct**


False 2k [0, 2, 4, ...] 100 True _\_ 39.8
False 2k [1, 3, 5, ...] 100 True _\_ 24.2
False 2k lazy 100 True _\_ 20.2
False 2k non-lazy 100 True _\_ **49.8**


16


