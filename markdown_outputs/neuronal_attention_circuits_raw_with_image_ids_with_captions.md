## **Neuronal Attention Circuit (NAC) for Representation Learning**

**Waleed Razzaq** [1] **Izis Kankaraway** [1] **Yun-Bo Zhao** [1 2]



**Abstract**

Attention improves representation learning over
RNNs, but its discrete nature limits continuoustime (CT) modeling. We introduce Neuronal Attention Circuit (NAC), a novel, biologically plausible CT-Attention mechanism that reformulates

attention logits computation as the solution to a
linear first-order ODE with nonlinear interlinked
gates derived from repurposing _C. elegans_ Neuronal Circuit Policies (NCPs) wiring mechanism.
NAC replaces dense projections with sparse sensory gates for key-query projections and a sparse
backbone network with two heads for computing
_content-target_ and _learnable time-constant_ gates,
enabling efficient adaptive dynamics. NAC supports three attention logit computation modes: (i)
explicit Euler integration, (ii) exact closed-form
solution, and (iii) steady-state approximation. To
improve memory intensity, we implemented a
sparse Top- _K_ pairwise concatenation scheme that
selectively curates key-query interactions. We
provide rigorous theoretical guarantees, including
state stability, bounded approximation errors, and
universal approximation. Empirically, we implemented NAC in diverse domains, including irregular time-series classification, lane-keeping for
autonomous vehicles, and industrial prognostics.
We observed that NAC matches or outperforms
competing baselines in accuracy and occupies an
intermediate position in runtime and memory efficiency compared with several CT baselines.


**1. Introduction**


Learning representations of sequential data in temporal or
spatio-temporal domains is essential for capturing patterns
and enabling accurate forecasting. Discrete-time Recurrent neural networks (DT-RNNs) such as RNN (Rumelhart et al., 1985; Jordan, 1997), Long-short term memory


1Department of Automation, University of Science & Technology of China, Hefei, China [2] Institute of Artificial Intelligence,
Hefei Comprehensive National Science Center. Correspondence to:
Yun-Bo Zhao _<_ ybzhao@ustc.edu.cn _>_, Waleed Razzaq _<_ waleedrazzaq@mail.ustc.edu.cn _>_ .


_Preprint. December 12, 2025._



(LSTM) (Hochreiter & Schmidhuber, 1997), and Gated Recurrent Unit (GRU) (Cho et al., 2014) model sequential dependencies by iteratively updating hidden states to represent
or predict future elements in a sequence. While effective
for regularly sampled sequences, DT-RNNs face challenges
with irregularly sampled data because they assume uniform
time intervals. In addition, vanishing gradients can make
it difficult to capture long-term dependencies (Hochreiter,
1998).
Continuous-time RNNs (CT-RNNs) (Rubanova et al., 2019)
model hidden states as ordinary differential equations
(ODEs), allowing them to process inputs that arrive at arbitrary or irregular time intervals. Mixed-memory RNNs
(mmRNNs) (Lechner & Hasani, 2022) build on this idea
by separating memory compartments from time-continuous
states, helping maintain stable error propagation while capturing continuous-time dynamics. Liquid neural networks
(LNNs) (Hasani et al., 2021; 2022) take a biologically inspired approach by assigning variable time-constants to hidden states, improving adaptability and robustness, though
vanishing gradients can still pose challenges during training.
The attention mechanisms (Vaswani et al., 2017) mitigate
this limitation by treating all time steps equally and allowing models to focus on the most relevant observations. It
computes the similarity between queries ( _q_ ) and keys ( _k_ ),
scaling by the key dimension to keep gradients stable. MultiHead Attention (MHA) (Vaswani et al., 2017) extends this
by allowing the model to attend to different representation
subspaces in parallel. Variants like Sparse Attention (Tay
et al., 2020; Roy et al., 2021), BigBird (Zaheer et al., 2020),
and Longformer (Beltagy et al., 2020) modify the attention
pattern to reduce computational cost, particularly for long sequences, by attending only to selected positions rather than
all pairs. Even with these improvements, attention-based
methods still rely on discrete scaled dot-product operations,
limiting their ability to model continuous trajectories often
captured by CT counterparts.
Recent work has explored bridging this gap through NeuralODE (Chen et al., 2018) formulation. mTAN (Shukla &
Marlin, 2021) learns CT embeddings and uses time-based
attention to interpolate irregular observations into a fixedlength representation for downstream encoder-decoder modeling. ODEFormer (d’Ascoli et al., 2023) trains a sequenceto-sequence transformer on synthetic trajectories to directly
infer symbolic ODE systems from noisy, irregular data,
though it struggles with chaotic systems and generalization



1


**Neuronal Attention Circuit (NAC) for Representation Learning**



beyond observed conditions. Continuous-time Attention
(CTA) (Chien & Chen, 2021) embeds a continuous-time
attention mechanism within a Neural ODE, allowing attention weights and hidden states to evolve jointly over time.
Still, it remains computationally intensive and sensitive to
the accuracy of the ODE solver. ContiFormer (Chen et al.,
2023) builds a CT-transformer by pairing ODE-defined latent trajectories with a time-aware attention mechanism to
model dynamic relationships in data.
Despite these innovations, a persistent and underexplored
gap remains in developing a biologically plausible attention
mechanism that seamlessly integrates CT dynamics with the
abstraction of the brain’s connectome to handle irregular sequences without prohibitive computational costs. Building
on this, we propose a novel attention mechanism called the
_Neuronal Attention Circuit_ (NAC), in which attention logits
are computed as the solution to a first-order ODE modulated
by nonlinear, interlinked gates derived from repurposing
Neuronal Circuit Policies (NCPs) from the nervous system
of _C. elegans_ nematode (refer to Appendix A.2 for more
information). Unlike standard attention, which projects keyquery pairs through a dense layer, NAC employs a sensory
gate to transform input features and a backbone to model
nonlinear interactions, with multiple heads producing outputs structured for attention logits computation. Based on
the solutions to ODE, we define three computation modes:
(i) Exact, using the closed-form ODE solution; (ii) Euler,
approximating the solution via _explicit Euler_ integration;
and (iii) Steady, using only the steady-state solution, analogous to standard attention scores. To reduce computational
complexity, we implemented a sparse Top- _K_ pairwise concatenation algorithm that selectively curates key-query inputs. We evaluate NAC across multiple domains, including
irregularly sampled time series, autonomous vehicle lanekeeping, and Industry 4.0, comparing it to state-of-the-art
baselines. NAC consistently matches or outperforms these
models, while runtime and peak memory benchmarks place
it between CT-RNNs in terms of speed and CT-Attentions
in terms of memory requirements.


**2. Neuronal Attention Circuit (NAC)**


We propose a simple alternative formulation of the attention
logits _a_ (refer to Appendix A.1 for more information), interpreting them as the solution to a first-order linear ODE
modulated by nonlinear, interlinked gates:



from repurposing NCPs. We refer to this formulation as
the Neuronal Attention Circuit (NAC). It enables the logits
_at_ to evolve dynamically with input-dependent, variable
time constants, mirroring the adaptive temporal dynamics
found in _C. elegans_ nervous systems while improving
computational efficiency and expressiveness. Moreover, it
introduces continuous depth into the attention mechanism,
bridging discrete-layer computation with dynamic temporal
evolution.

**Motivation behind this formulation:** The proposed
formulation is loosely motivated by the input-dependent
time-constant mechanism of Liquid Neural Networks
(LNNs), a class of CT-RNNs inspired by biological nervous
systems and synaptic transmission. In this framework, the
dynamics of non-spiking neurons are described by a linear
ODE with nonlinear, interlinked gates: _ddt_ **xt** = **[x]** _τ_ **[t]** [+] **[ S][t]** _[,]_

where **xt** denotes the hidden state and **St** _∈_ R _[M]_ represents
a nonlinear contribution defined as _f_ ( **xt** _,_ **u** _, t, θ_ )( _A −_ **xt** ).
Here, _A_ and _θ_ are learnable parameters. Plugging **St**
yields _[d]_ **[x][t]** [=] _[ −]_ � _−_ [1] _[−]_ _[f]_ [(] **[x][t]** _[,]_ **[ u]** _[, t, θ]_ [)] � **xt** + _f_ ( **xt** _,_ **u** _, t, θ_ ) _A_ .



_an_ +1 = _an_ + ∆ _t_ ( _−ωτ_ _an_ + _ϕ_ ) _._ (2)


**Closed-form (Exact) Computation of NAC:** We now devise the analytical solution for Eqn. 1. Let both _ωτ_ and
_ϕ_ be fixed in pseudo-time interval (frozen-coefficient approximation (John, 1952)) with initial condition _a_ 0, then
closed-form solution is:



**xt** = **[x][t]**

_dt_ _τ_




**[x][t]** _−_ [1]

_dt_ [=] _[ −]_ � _τ_



yields _[d]_ _dt_ **[x][t]** [=] _[ −]_ � _−_ _τ_ [1] _[−]_ _[f]_ [(] **[x][t]** _[,]_ **[ u]** _[, t, θ]_ [)] � **xt** + _f_ ( **xt** _,_ **u** _, t, θ_ ) _A_ .

LNNs are known for their strong expressivity, stability, and
performance in irregularly sampled time-series modeling
(Hasani et al., 2021; 2022).
**NAC’s forward-pass update using ODE solver:** The
state of NAC at time _t_ can be computed using a numerical
ODE solver that simulates the dynamics from an initial
state _a_ 0 to _at_ . The solver discretizes the continuous interval

[0 _, T_ ] into steps [ _t_ 0 _, t_ 1 _, t_ 2 _, . . ., tn_ ], with each step updating
the state from _ti_ to _ti_ +1. For our purposes, we use the
_explicit Euler_ solver, which is simple, efficient, and easy to
implement. Although methods such as Runge-Kutta may
offer higher accuracy, their computational overhead makes
them less suitable for large-scale neural simulations that
require numerous updates, especially since the logits are
normalized, and exact precision is not necessary. Let the
step size be ∆ _t_, with discrete times _tn_ = _n_ ∆ _t_ and logit
states _an_ = _a_ ( _tn_ ). Using the _explicit Euler_ method, the
update is



_dat_

_dt_ [=] _[ −]_ ~~�~~ _[f][ω][τ]_ [ ([] **[q]** � [;] **[ k]** ~~�~~ []] _[, θ][ω][τ]_ [ )] ~~�~~
_ωτ_ ( **u** )



_at_ + _fϕ_ ([ **q** ; **k** ] _, θϕ_ ) _,_ (1)
� ~~�~~ � ~~�~~
_ϕ_ ( **u** )



_at_ = _a_ _[∗]_ + ( _a_ 0 _−_ _a_ _[∗]_ ) _e_ _[−][ω][τ][ t]_
���� ~~�~~ ~~��~~ ~~�~~
steady-state transient



(3)



where **u** = [ **q** ; **k** ] denotes the sparse Top- _K_ concatenated
query–key input. _ωτ_ represents a learnable time-constant
gate head, _ϕ_ denotes a nonlinear content-target head. Both
gates are parameterized by a backbone network derived



Here, _a_ _[∗]_ = _ϕ/ωτ_ is the steady-state solution. The full
derivation is provided in Appendix B.1.



2


**Neuronal Attention Circuit (NAC) for Representation Learning**



**2.1. Stability Analysis of NAC**


We now investigate the stability bounds of _NAC_ under both
the ODE-based and the Closed-Form formulations.


2.1.1. STATE STABILITY


We analyze state stability in both single-connection and
multi-connection settings. This analysis establishes the
boundedness of the attention logit state trajectory, ensuring that, under positive decay rates, the dynamics remain
well-behaved without divergence or overshoot.

**Theorem 1** (State Stability) **.** _Let a_ [(] _t_ _[i]_ [)] _denote the state of_
_the i-th attention logit governed by da_ [(] _t_ _[i]_ [)] _[/dt]_ [ =] _[ −][ω][τ]_ _[a]_ [(] _t_ _[i]_ [)] +
_ϕ. Assume that ϕ and ωτ decompose across M incom-_
_ing connections as ϕ_ = [�] _[M]_ _j_ =1 _[f][ϕ]_ [([] **[q]** _[i]_ [;] **[ k]** _[j]_ [])] _[, and][ ω][τ]_ [ =]
_M_
� _j_ =1 _[f][ω]_ _τ_ [([] **[q]** _[i]_ [;] **[ k]** _[j]_ [])] _[,][ with][ f][ω]_ _τ_ _>_ 0 _._ _Define the per-_
_connection equilibrium Ai,j_ = _fϕ_ ([ **q** _i_ ; **k** _j_ ]) _/fωτ_ ([ **q** _i_ ; **k** _j_ ]) _,_
_and let A_ [min] _i_ = min _j Ai,j and A_ [max] _i_ = max _j Ai,j. Then_
_for any finite horizon t ∈_ [0 _, T_ ] _, the state trajectory satisfies_


min(0 _, A_ [min] _i_ ) _≤_ _a_ [(] _t_ _[i]_ [)] _≤_ max(0 _, A_ [max] _i_ ) _,_ (4)


_provided the initial condition ai_ (0) _lies within this range. In_
_the special case of a single connection (M_ = 1 _), the bounds_
_collapse to_


min(0 _, Ai_ ) _≤_ _a_ [(] _t_ _[i]_ [)] _≤_ max(0 _, Ai_ ) _,_ (5)


_where Ai_ = _fϕ/ωτ is exactly the steady-state solution from_
_Eqn. 3. The proof is provided in the Appendix B.2._


2.1.2. CLOSED-FORM ERROR & EXPONENTIAL BOUNDS


We now examine the asymptotic stability, error characterization, and exponential boundedness of the closed-form
formulation. We begin by quantifying the deviation of the
trajectory from its steady-state solution. Define the instanta
neous error
_εt_ = _at −_ _a_ _[∗]_ _,_ (6)


which measures the distance of the system state to equilibrium at time _t_ . From Eqn. 3, the error admits the exact
representation


_εt_ = ( _a_ 0 _−_ _a_ _[∗]_ ) _e_ _[−][ω][τ][ t]_ (7)


In particular, the pointwise absolute error is given by


_|εt|_ = _|a_ 0 _−_ _a_ _[∗]_ _| e_ _[−][ω][τ][ t]_ (8)


This reveals that convergence is not merely asymptotic but
follows an exact exponential law, controlled by the rate parameter _ωτ_ . This yields the following finite-time guarantee.

**Corollary 1** (Exponential decay bound) **.** _If ωτ >_ 0 _, then_
_for all t ≥_ 0 _,_


_|at −_ _a_ _[∗]_ _| ≤|a_ 0 _−_ _a_ _[∗]_ _| e_ _[−][ω][τ][ t]_ _._ (9)



_Remark:_ If _ωτ >_ 0 then lim _t→∞_ _e_ _[−][ω][τ][ t]_ = 0, therefore
_at →_ _a_ _[∗]_ . The convergence is exponential with rate _ωτ_ .
If _ωτ <_ 0 then _e_ _[−][ω][τ][ t]_ = _e_ _[|][ω][τ][ |][t]_ diverges so _at_ grows
exponentially away from _a_ _[∗]_ in magnitude (unless initial
offset _a_ 0 _−_ _a_ _[∗]_ = 0, a measure-zero case). If _ωτ_ = 0 the
ODE is ˙ _a_ = _ϕ_ and the solution is linear in _t_ (unless _ϕ_ = 0).
For bounded dynamics that converge to an interpretable
steady-state, it is required that _ωτ >_ 0.


**Corollary 2** (Uniform initialization) **.** _If the initialization is_
_only known to belong to a bounded set, i.e., |a_ 0 _−_ _a_ _[∗]_ _| ≤_ _M_
_for some M >_ 0 _, then the error admits the uniform bound_


_|at −_ _a_ _[∗]_ _| ≤_ _Me_ _[−][ω][τ][ t]_ _._ (10)


_Remark:_ This bound highlights that exponential convergence holds uniformly across all admissible initial
conditions, with the constant _M_ capturing the worst-case
deviation.


**Corollary 3** (Sample complexity to _δ_ -accuracy) **.** _A natural_
_operational question is the time required to achieve a target_
_tolerance δ >_ 0 _. Solving_


_|a_ 0 _−_ _a_ _[∗]_ _|e_ _[−][ω][τ][ t]_ _≤_ _δ,_ (11)


_We obtain the threshold_




[1] ln _[|][a]_ [0] _[ −]_ _[a][∗][|]_

_ωτ_ _δ_



_t ≥_ [1]



_._ (12)
_δ_



_Remark:_ The convergence rate is inversely proportional to
_ωτ_, and the required time scales only logarithmically in
the accuracy level 1 _/δ_ . Intuitively, larger _ωτ_ accelerates
contraction towards equilibrium, yielding faster attainment
of any prescribed tolerance.


_Figure 1._ Illustration of **(a)** NCPs with pre-determined wiring; **(b)**
Sensory gate, where sensory neurons are active, and the remaining
neurons are disabled for the _q_, _k_, and _v_ projections; **(c)** Backbone,
showing inter-motor projections with sensory neurons disabled in
extended heads for computing _ϕ_ and _ωτ_ .



3




```json
"img_neuronal_attention_circuits_2_0": {
    "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/neuronal_attention_circuits.pdf-2-0.png",
    "page": 2,
    "section": "Abstract",
    "image_relevance": "high",
    "image_type": "architecture",
    "semantic_role": "illustrates",
    "caption": "The image presents three distinct neural circuit architectures: a) Neuronal Circuit Policies, b) Sensory Gate, and c) Backbone. Each circuit is composed of sensory (dark purple triangle), inter (blue circle), command (purple square), and motor (pink triangle) neurons, with some command neurons exhibiting feedback loops. The diagrams also delineate 'active group' (light green) and 'disabled group' (light grey) components, illustrating variations in neuron activation and connectivity from input to output.",
    "depicted_concepts": [
      "Neuronal circuit",
      "Sensory neuron",
      "Interneuron",
      "Command neuron",
      "Motor neuron",
      "Neural network architecture",
      "Feedback loop",
      "Sensory Gate",
      "Backbone circuit",
      "Active group",
      "Disabled group"
    ],
    "confidence": "high"
}
```
**Neuronal Attention Circuit (NAC) for Representation Learning**



**Algorithm 1** Repurposed NCPCell


**Require:** Wiring _W_ with ( _A_ in _, A_ rec), groups ( _Ns, Ni, Nc, Nm_ ),
activation _α_, input group _G_ input, output group _G_ output, disabled
groups _D_
**Ensure:** Output _yt ∈_ R _[B][×][d]_ [out], state **x** _t ∈_ R _[B][×][d][h]_

Binary mask: _M_ rec _←|A_ rec _|_, _M_ in _←|A_ in _|_
Initialize parameters : _W_ in _, W_ rec _, b, w_ in _, b_ in _, w_ out _, b_ out
Input neurons: _I_ in _←_ _G_ input
Output neurons: _I_ out _←_ _G_ output
Define activation mask: maskact _,i_ = 0 if _i ∈D_ else 1
Input Projections: ˜ _ut ←_ _ut ⊙_ _w_ in + _b_ in
Recurrent computation:Sparse computation: _st ← rt ←u_ ˜ _t_ ( **x** _Wt−_ in1 _⊙_ ( _WM_ recin _⊙_ ) _M_ rec)
Neuron update: **x** _t ←_ _α_ ( _rt_ + _st_ + _b_ ) _⊙_ maskact
Output mapping: _yt ←_ ( **x** _t_ [ _I_ out] _⊙_ _w_ out) + _b_ out
**return** ( _yt,_ **x** _t_ )


**Algorithm 2** Sparse Top- _K_ Pairwise Concatenation


**Require:** Keys _K ∈_ R _[B][×][H][×][T][k][×][D]_, Top- _K_ value _K_
**Ensure:** concatenated tensor _U ∈_ R _[B][×][H][×][T][q]_ _[×][K]_ [eff] _[×]_ [2] _[D]_

Scores: _S ←_ _Q · K_ _[⊤]_

Effective Top- _K_ : _K_ eff _←_ min( _K, Tk_ )
Indices: _I_ topk _←_ top ~~k~~ ( _S, K_ eff)
Gather: _K_ selected _←_ gather( _K, I_ topk) _∈_ R _[B][×][H][×][T][q]_ _[×][K]_ [eff] _[×][D]_

Tiled: _Q_ tiled _←_ tile( _Q, K_ eff) _∈_ R _[B][×][H][×][T][q]_ _[×][K]_ [eff] _[×][D]_

Concatenate: _U_ topk _←_ [ _Q_ tiled; _K_ selected ] _∈_ R _[B][×][H][×][T][q]_ _[×][K]_ [eff] _[×]_ [2] _[D]_

**return** _U_ topk


**2.2. Designing the Neural Network**


We now outline the design of a neural network layer guided
by the preceding analysis. The process involves five steps:
(i) repurposing NCPs; (ii) input curation; (iii) construction
of the time vector ( _t_ ); (iv) computing attention logits and
weights; and (v) generating the attention output. Figure 2
provides a graphical overview of NAC.
**Repurposing NCPs:** We repurpose the NCPs framework
by converting its fixed, biologically derived wiring (see Figure 1(a)) into a flexible recurrent architecture that allows
configurable input–output mappings. Instead of enforcing
a static connectome, our approach exposes adjacency matrices as modifiable structures defining sparse input and
recurrent connections. This enables selective information

routing across neuron groups while retaining the original circuit topology. Decoupling wiring specifications from model
instantiation allows dynamic connectivity adjustments to
accommodate different input modalities without full retraining. Algorithm 1 summarizes the steps for repurposing the
NCPs wiring mechanism. Key features include group-wise
masking for neuron isolation, adaptive remapping of inputs
and outputs for task-specific adaptation, and tunable sparsity
_s_ to balance expressiveness and efficiency.
In our implementation, the sensory neuron gate ( _NN_ sensory)
projects the _q_, _k_, and _v_ representations (see Figure 1(b)).
This enables sensory neurons to maintain structured, contextaware representations rather than collapsing inputs into fully



connected layers. As a result, the network preserves locality
and modularity, which improves information routing.


_NN_ sensory = NCPCell( _G_ input = [ _Ns_ ] _, G_ output = [ _Ns_ ] _,_

_D_ = [ _Ni, Nc, Nm_ ] _, s_ )
(13)
The inter-to-motor pathways form a backbone network
( _NN_ backbone) with branches that compute _ϕ_ and _ωτ_ (see
Figure 1(c)). Instead of learning _ϕ_ and _ωτ_ independently,
this backbone allows the model to learn shared representations, enabling multiple benefits: (i) separate head layers
enable the system to capture temporal and structural dependencies independently; (ii) accelerates convergence during
training.


_NN_ backbone = NCPCell( _G_ input = [ _Ni_ ] _, G_ output = [ _Nm_ ] _,_

_D_ = [ _Ns_ ] _, s_ )
(14)
The output heads are defined as:


_ϕ_ = _σ_ ( _NN_ backbone( **u** )) (15)

_ωτ_ = softplus( _NN_ backbone( **u** )) + _ε,_ _ε >_ 0 (16)


Here, _ϕ_ serves as a _content–target gate_ head, where the
sigmoid function _σ_ ( _·_ ) determines the target signal strength.
In contrast, _ωτ_ is a strictly positive _time–constant gate_ head
that controls the rate of convergence and the steady-state
amplitude. Conceptually, this parallels recurrent gating: _ϕ_
regulates _what_ content to emphasize, while _ωτ_ governs _how_
_quickly_ and _to what extent_ it is expressed.
**Input Curation:** We experimented with different
strategies for constructing query–key inputs. Initially, we implemented full pairwise concatenation,
where queries _Q ∈_ R _[B][×][H][×][T][q][×][D]_ are combined with
all keys _K_ _∈_ R _[B][×][H][×][T][k][×][D]_ to form a joint tensor
_U ∈_ R _[B][×][H][×][T][q][×][T][k][×]_ [2] _[D]_ . While this preserved complete
feature information and enabled expressive, learnable
similarity functions, it was memory-intensive, making it impractical for longer sequences. To mitigate this, we applied
a sparse Top- _K_ optimization: for each query, we compute
pairwise scores _S_ = _Q · K_ _[⊤]_ _∈_ R _[B][×][H][×][T][q][×][T][k]_, select the
Top- _K_ eff = min( _K, Tk_ ) keys, and construct concatenated
pairs _U_ topk _∈_ R _[B][×][H][×][T][q][×][K]_ [eff] _[×]_ [2] _[D]_ . This approach preserves
the most relevant interactions while substantially reducing
memory requirements in the concatenation and subsequent
backbone processing stages, allowing the method to scale
linearly with the sequence length in those components.
However, the initial computation of _S_ remains quadratic
(see Appendix C.3). Algorithm 2 outlines the steps required
for input curation.
**Time Vector:** NAC builds on continuous-depth models
as (Hasani et al., 2022) that adapt their temporal dynamics to the task. It constructs an internal, normalized
pseudo-time vector _t_ pseudo using a sigmoidal transformation,
_t_ pseudo = _σ_ ( _ta · t_ + _tb_ ), where _ta_ and _tb_ are learnable affine



4


**Neuronal Attention Circuit (NAC) for Representation Learning**



parameters and _σ_ is the sigmoid function. For time-varying
datasets (e.g., irregularly sampled series), each time point
_t_ is derived from the sample’s timestamp, while for tasks
without meaningful timing, _t_ is set to 1. The resulting _t_ pseudo
lies in [0 _,_ 1] and provides a smooth, bounded representation
of time for modulating the network’s dynamics.
**Attention logits and weights:** Starting from Eqn. 3,
consider the trajectory of a query–key pair with initial
condition _a_ 0 = 0:



_at_ = _[ϕ]_

_ωτ_



�1 _−_ _e_ _[−][ω][τ][ t]_ [�] _,_ (17)



followed by the _softmax_ normalization to calculate attention weights. The resulting attention weights _αt_ [(] _[h]_ [)] are then
used to integrate with the value vector _v_ [(] _[h]_ [)], producing headspecific attention outputs. Finally, these outputs are concatenated and linearly projected back into the model dimension.
This formulation ensures that each head learns distinct dynamic compatibilities governed by its own parameterization
of _ϕ_ and _ωτ_, while the aggregation across heads preserves
the expressive capacity of the standard multi-head attention
mechanism.


**2.3. NAC as Universal Approximator**


We now establish the universal approximation capability of
NAC by extending the classical Universal Approximation
Theorem (UAT) (Nishijima, 2021) to the proposed mechanism. For brevity, we consider a network with a single
NAC layer processing fixed-dimensional inputs, though the
argument generalizes to sequences.


**Theorem 2** (Universal Approximation by NAC) **.** _Let K ⊂_
R _[n]_ _be a compact set and f_ : _K →_ R _[m]_ _be a continuous_
_function. For any ϵ >_ 0 _, there exists a neural network_
_consisting of a single NAC layer, with sufficiently large_
_model dimension dmodel, number of heads H, sparsity s,_
_and nonlinear activations, such that the network’s output_
_g_ : R _[n]_ _→_ R _[m]_ _satisfies_


sup _∥f_ ( _x_ ) _−_ _g_ ( _x_ ) _∥_ _< ϵ._ (20)
_x∈K_


_The proof is provided in Appendix B.3._


**3. Evaluation**


We evaluate the proposed architecture against a range of
baselines, including (DT & CT) RNN, (DT & CT) attention,
and multiple NAC ablation configurations. Experiments
are conducted across diverse domains, including irregular
time-series modeling, lane keeping of autonomous vehicles,
and Industry 4.0 prognostics. All results are obtained using 5-fold cross-validation, where models are trained using
BPTT (see Appendix C.2) on each fold and evaluated across
all folds. We report the mean ( _µ_ ) and standard deviation ( _σ_ )
to capture variability and quantify uncertainty in the predictions. Table 1 provides results for all experiments, and the
details of the baselines, ablation, environment utilized, the
data curation and preprocessing, and neural network architectures for all experiments are provided in the Appendix
D.3.


**3.1. Irregular Time-series**


We evaluate the proposed architecture on two irregular timeseries datasets: (i) Event-based MNIST; and (ii) Person
Activity Recognition (PAR).



For finite _t_, the exponential factor (1 _−_ _e_ _[−][ω][τ][ t]_ ) regulates the
buildup of attention, giving _ωτ_ a temporal gating role. Normalizing across all keys via _softmax_ yields attention weights
_αt_ = softmax( _at_ ), defining a valid probability distribution
where _ϕ_ amplifies or suppresses content alignments, and _ωτ_
shapes both the speed and saturation of these preferences.
As _t →∞_, the trajectory converges to the steady state


_a_ _[∗]_ _t_ [=] _[ϕ]_ _≈_ _[q][⊤][k]_ _,_ (18)

_ωτ_ ~~_√_~~ _dk_


which is analogous to scaled-dot attention under specific
parameterization when the backbone _NN_ backbone is configured as a linear projection such that _ϕ_ ( **u** ) = _q_ _[⊤]_ _k_ and
_ωτ_ ( _u_ ) = _[√]_ _dk_ (e.g., by setting NCP weights to emulate
bilinear forms and disabling nonlinearities). In general, the
nonlinear backbone allows for more expressive similarities,
with the approximation holding when trained to mimic dot
products.
**Attention output:** Finally, the attention output is computed
by integrating the attention weights with the value matrix:


NAC( _q, k, v_ ) = _αtvtdt_ (19)
� _T_


In practice, the integration is approximated using a Riemannstyle approach, where the weighted elements are computed
by multiplying each _vt_ with its corresponding _αt_ . These are
then summed and multiplied by a fixed pseudo-time step
_δt_, chosen as a scalar (typically between 0.5–1.0) hyperparameter during layer initialization. This yields a continuous
analogue of standard weighted sums, giving finer resolution
of the attention trajectory without altering the underlying
values. Sensitivity to attention output w.r.t _δt_ is visualized
in Appendix D.2.


2.2.1. EXTENSION TO MULTI-HEAD


To scale this mechanism to multi-head attention, we project
the input sequence into _H_ independent subspaces (heads)
of dimension _d_ model _/H_, yielding query, key, and value tensors ( _q_ [(] _[h]_ [)] _, k_ [(] _[h]_ [)] _, v_ [(] _[h]_ [)] ) for _h ∈{_ 1 _, . . ., H}_ . For each head,
pairwise logits are computed according to Eqns. 2,3 or 18,



5


**Neuronal Attention Circuit (NAC) for Representation Learning**


_Figure 2._ Illustration of the architecture of **(a)** Neuronal Attention Circuit mechanism ; **(b)** Multi-Head Extension




```json
"img_neuronal_attention_circuits_5_0": {
    "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/neuronal_attention_circuits.pdf-5-0.png",
    "page": 5,
    "section": "Abstract",
    "image_relevance": "high",
    "image_type": "architecture",
    "semantic_role": "defines",
    "caption": "The image presents two architectural diagrams: a Neuronal Attention Circuit (NAC) and a Multi-Head Neuronal Attention Circuit. The single-head NAC processes time-series Query, Key, and Value inputs through Sparse Topk-Pairwise, a Backbone module, a logit computation step, softmax, and an Attention Score function to yield NAC(q,k,v). The Multi-Head NAC extends this by channeling Query, Key, and Value inputs through Sensory gates into multiple instances of the Neuronal Attention Circuit, whose outputs are then concatenated, transformed by a linear layer, and subjected to an activation function.",
    "depicted_concepts": [
      "Neuronal Attention Circuit (NAC)",
      "Multi-Head Neuronal Attention Circuit",
      "Query",
      "Key",
      "Value",
      "Sparse Topk-Pairwise",
      "Backbone",
      "Compute logits",
      "softmax",
      "Attention Score",
      "Sensory gate",
      "Concatenation",
      "Linear layer",
      "Activation function"
    ],
    "confidence": "high"
}
```

**Event-based MNIST:** Event-based MNIST is the trans
formation of the widely recognized MNIST dataset with
irregular sampling added originally proposed in (Lechner
& Hasani, 2022). The transformation was done in two
steps: (i) flattening each 28×28 image into a time series
of length 784, and (ii) encoding the binary time series into
an event-based format by collapsing consecutive identical
values (e.g., 1,1,1,1 → (1, t=4)). This representation requires models to handle temporal dependencies effectively.
NAC-PW achieved first place with an accuracy of 96.64%,
followed by NAC-Exact/05s/8k at 96.12%. GRU-ODE and
ContiFormer ranked third with 96.04%.

**Person Activity Recognition (PAR):** We employed the
Localized Person Activity dataset from UC Irvine (Vidulin
et al., 2010). The dataset contains data from five participants,
each equipped with inertial measurement sensors sampled
every 211 ms. The goal of this experiment is to predict a
person’s activity from a set of predefined actions, making it
a classification task. All models performed well on this task,
with NAC-PW achieving 89.15% accuracy and taking first
place. NAC-Exact/05s/8k and GRU-ODE ranked second
with 89.01% accuracy, while NAC-02s ranked third with
88.84% mean accuracy.


**3.2. Lane-Keeping of Autonomous Vehicles**


Lane keeping in autonomous vehicles (AVs) is a fundamental problem in robotics and AI. Early works (Tiang et al.,
2018; Park et al., 2021) primarily emphasized accuracy,
often relying on large models. More recent research (Lechner et al., 2020; Razzaq & Hongwei, 2023) has shifted toward designing compact architectures suitable for resourceconstrained devices. The goal of this experiment is to create
a long causal structure between the road’s horizon and the



corresponding steering commands. To evaluate, we used
two widely adopted simulation environments: (i) OpenAI
CarRacing (Brockman et al., 2016); and (ii) the Udacity SelfDriving Car Simulator (uda). In OpenAI CarRacing, the
task is to classify steering actions from a predefined action
set. In contrast, the Udacity Simulator requires predicting a
continuous trajectory of steering values. We implemented
the AI models proposed by (Razzaq & Hongwei, 2023),
replacing the recurrent layer with NAC and its counterparts.
All models achieved around 80% accuracy on average in
the CarRacing benchmark. Notably, NAC-PW performed
the best, reaching the highest accuracy of 80.72%, followed
by NAC-Steady, ranked second with 80.62%. LSTM and
GRU took third position, achieving 80.60% on average.
In the Udacity benchmark, NAC-32k performed the best,
achieving the lowest MSE of 0.0170. NAC-Exact followed
with 0.0173, and ContiFormer ranked third with 0.0174.
To visualize saliency maps for these experiments, refer to
Appendix D.3.3. Experimental videos are available for the
[OpenAI CarRacing [click here] and for the Udacity Simula-](https://www.youtube.com/watch?v=kwTNU8aV8-I)
[tor [click here].](https://www.youtube.com/watch?v=mMRVsNUQ8i0)


**3.3. Industry 4.0**


Industry 4.0 has transformed manufacturing, making prognostic health management (PHM) systems essential. A key
PHM task is estimating the remaining useful life (RUL) of
components, particularly rolling element bearings (REB),
which account for 40–50% of machine failures (Ding et al.,
2021; Zhuang et al., 2021). The objective is to learn degradation features from one operating condition of a dataset
and generalize to unseen conditions within the same dataset.
Furthermore, the model should provide accurate RUL estimation on entirely different datasets, while maintaining a



6


**Neuronal Attention Circuit (NAC) for Representation Learning**


_Table 1._ Model Performance Across All Categories and Datasets


**Irregular Time-Series** **Lane-Keeeping of AVs** **Industry 4.0**
**Model**

**E-MNIST (↑)** **PAR (↑)** **CarRacing (↑)** **Udacity (↓)** **PRONOSTIA (↓)** **XJTU-SY (↓)** **HUST (↓)**


RNN 95.59 _[±]_ [0.37] 88.77 _[±]_ [0.58] 78.90 _[±]_ [3.35] 0.0210 _[±]_ [0.0014] 42.05 _[±]_ [7.49] 31.07 _[±]_ [6.63] 42.22 _[±]_ [8.16]

LSTM 95.88 _[±]_ [0.23] 88.36 _[±]_ [0.79] 80.60 _[±]_ [0.12] 0.0181 _[±]_ [0.0014] 41.87 _[±]_ [2.88] 31.99 _[±]_ [8.32] 44.09 _[±]_ [2.14]

GRU 95.85 _[±]_ [0.22] 88.68 _[±]_ [1.35] 80.60 _[±]_ [0.22] 0.0206 _[±]_ [0.0014] 44.22 _[±]_ [4.60] 26.65 _[±]_ [4.49] 41.86 _[±]_ [7.96]

CT-RNN 95.18 _[±]_ [0.20] 88.71 _[±]_ [0.87] 80.21 _[±]_ [0.27] 0.0206 _[±]_ [0.0013] 44.32 _[±]_ [8.69] 26.01 _[±]_ [8.74] 39.99 _[±]_ [6.33]

GRU-ODE **96.04** _[±]_ **[0.13]** **89.01** _[±]_ **[1.55]** 80.29 _[±]_ [0.72] 0.0188 _[±]_ [0.0016] 45.11 _[±]_ [3.19] 31.20 _[±]_ [8.69] 43.91 _[±]_ [7.10]

PhasedLSTM 95.79 _[±]_ [0.14] 88.93 _[±]_ [1.08] 80.35 _[±]_ [0.38] 0.0186 _[±]_ [0.0015] 44.15 _[±]_ [4.80] 35.49 _[±]_ [5.54] 38.66 _[±]_ [5.55]

mmRNN 95.74 _[±]_ [0.27] 88.48 _[±]_ [0.46] 80.13 _[±]_ [0.54] 0.0205 _[±]_ [0.0027] 48.50 _[±]_ [4.60] 27.84 _[±]_ [4.05] 40.11 _[±]_ [9.56]

LTC 95.25 _[±]_ [0.00] 88.12 _[±]_ [0.68] 76.37 _[±]_ [3.01] 0.0245 _[±]_ [0.0024] 48.14 _[±]_ [5.01] 36.83 _[±]_ [8.57] 61.82 _[±]_ [15.64]

CfC 94.16 _[±]_ [0.49] 88.60 _[±]_ [0.34] 80.59 _[±]_ [0.33] 0.0198 _[±]_ [0.0022] 47.78 _[±]_ [3.54] 35.51 _[±]_ [3.94] 54.09 _[±]_ [10.13]

Attention 95.68 _[±]_ [0.23] 88.29 _[±]_ [0.98] 80.40 _[±]_ [0.26] 0.0193 _[±]_ [0.0009] 41.89 _[±]_ [6.98] 26.29 _[±]_ [4.06] 40.28 _[±]_ [4.23]

MHA 95.94 _[±]_ [0.15] 88.36 _[±]_ [1.06] 79.99 _[±]_ [0.49] 0.0185 _[±]_ [0.0017] 45.36 _[±]_ [5.16] 37.31 _[±]_ [12.20] 41.40 _[±]_ [7.72]

mTAN 95.97 _[±]_ [0.25] 88.08 _[±]_ [0.94] 80.86 _[±]_ [0.22] 0.0178 _[±]_ [0.0005] 44.41 _[±]_ [7.15] 41.34 _[±]_ [3.72] 66.29 _[±]_ [4.25]

CTA 95.86 _[±]_ [0.14] 88.10 _[±]_ [1.10] 80.54 _[±]_ [0.40] 0.0197 _[±]_ [0.0016] 39.16 _[±]_ [3.54] **25.86** _[±]_ [1.47] 38.41 _[±]_ [4.51]

ODEFormer 95.62 _[±]_ [0.20] 88.25 _[±]_ [0.66] 80.54 _[±]_ [0.40] 0.0190 _[±]_ [0.0012] 42.42 _[±]_ [6.98] 35.63 _[±]_ [9.24] 40.60 _[±]_ [6.83]

ContiFormer **96.04** _[±]_ **[0.23]** 81.28 _[±]_ [0.85] 80.47 _[±]_ [0.50] **0.0174** _[±]_ **[0.01]** **27.82** _[±]_ **[7.09]** 34.71 _[±]_ [4.98] 43.81 _[±]_ [10.18]


NAC-2k 95.73 _[±]_ [0.07] 88.84% _[±]_ [0.81] 80.59 _[±]_ [0.46] 0.0208 _[±]_ [0.0015] 43.78 _[±]_ [2.71] 37.43 _[±]_ [9.28] 40.51 _[±]_ [6.61]

NAC-32k 95.15 _[±]_ [0.11] 88.80 _[±]_ [0.76] 80.38 _[±]_ [0.16] **0.0170** _[±]_ **[0.0007]** 49.53 _[±]_ [4.89] 32.45 _[±]_ [10.84] 39.17 _[±]_ [12.23]

NAC-PW **96.64** _[±]_ [0.12] **89.15** _[±]_ **[1.01]** **80.72** _[±]_ **[0.41]** 0.0177 _[±]_ [0.0008] **37.50** _[±]_ **[2.56]** 28.01 _[±]_ [4.93] **30.14** _[±]_ **[6.87]**


NAC-FC 95.31 _[±]_ [0.07] 88.45 _[±]_ [0.91] 80.49 _[±]_ [0.46] 0.0192 _[±]_ [0.0012] 40.36 _[±]_ [6.09] **24.89** _[±]_ **[5.30]** **35.35** _[±]_ **[6.64]**

NAC-02s 95.31 _[±]_ [0.07] 88.84 _[±]_ [1.33] 80.47 _[±]_ [0.27] 0.0188 _[±]_ [0.0013] 39.43 _[±]_ [5.94] 35.59 _[±]_ [3.86] 38.90 _[±]_ [6.43]

NAC-09s 95.86 _[±]_ [0.11] 88.61% _[±]_ [1.25] 80.43% _[±]_ [0.17] 0.0188 _[±]_ [0.0013] 47.29 _[±]_ [5.52] 40.40 _[±]_ [8.85] 44.39 _[±]_ [6.82]


NAC-Exact/05s/8k **96.12** _[±]_ [0.11] **89.01** _[±]_ **[1.01]** 80.59 _[±]_ [1.82] **0.0173** _[±]_ **[0.0006]** **37.75** _[±]_ **[4.72]** **19.87** _[±]_ **[1.75]** **27.82** _[±]_ **[7.09]**

NAC-Euler 95.67 _[±]_ [0.26] 88.52 _[±]_ [0.68] **80.61** _[±]_ [0.28] 0.0181 _[±]_ [0.0017] 42.08 _[±]_ [6.14] 28.46 _[±]_ [8.18] 39.32 _[±]_ [9.15]

NAC-Steady 95.75 _[±]_ [0.28] 88.36 _[±]_ [1.05] **80.62** _[±]_ **[0.26]** 0.0181 _[±]_ [0.0012] 40.95 _[±]_ [5.77] 26.76 _[±]_ [7.36] 37.12 _[±]_ [12.43]


**Note:** (↑) higher is better; (↓) lower is better.


the lowest score of 27.82. NAC-Exact/05s/8k and NAC-PW

achieved nearly identical scores, obtaining 37.75 and 37.50
on average, respectively. On the XJTU-SY dataset, NACExact/05s/8k has the lowest score of 19.87. NAC-FC ranked

second with a score of 24.89, followed by NAC-PW in third
place with an average score of 28.01. A similar trend was
observed on the HUST dataset, where NAC-Exact/05s/8k
achieved first place with a score of 27.82, NAC-PW ranked
second with 30.14, and NAC-FC ranked third with 35.35.
These results demonstrated the strong cross-validation capability of NAC.




```json
"img_neuronal_attention_circuits_6_0": {
    "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/neuronal_attention_circuits.pdf-6-0.png",
    "page": 6,
    "section": "Abstract",
    "image_relevance": "high",
    "image_type": "plot",
    "semantic_role": "supports_result",
    "caption": "A line plot illustrates degradation estimation by comparing Normalized Degradation over Time for 'Expected' versus 'NAC' (Neural Architecture Search) curves across three distinct datasets: PRONOSTIA, XJTU-SY, and HUST. The plot shows the progression of degradation, ranging from 0.0 to 1.0, against time for each dataset's expected and NAC-estimated values.",
    "depicted_concepts": [
      "Degradation Estimation",
      "Normalized Degradation",
      "Time",
      "PRONOSTIA dataset",
      "XJTU-SY dataset",
      "HUST dataset",
      "Expected degradation",
      "NAC degradation"
    ],
    "confidence": "high"
}
```

_Figure 3._ Degradation Estimation Results.


compact architecture suitable for resource-constrained devices, thereby supporting localized safety.
We utilized three benchmark datasets: (i) PRONOSTIA
(Nectoux et al., 2012), (ii) XJTU-SY (Wang et al., 2018),
and (iii) HUST (Thuan & Hong, 2023). Training is performed on PRONOSTIA, while XJTU-SY and HUST are

used to assess cross-validation. We used the Score met
ric (Nectoux et al., 2012) to assess the performance. We
evaluate generalization using _Bearing 1_ from the first operating condition of each dataset. Figure 3 visualizes the
expected degradation alongside the outputs of NAC. On the
PRONOSTIA dataset, ContiFormer performed the best with



**3.4. Run Time and Memory Experiments**


We evaluate computational requirements on fixed-length
sequences of 1024 steps, 64-dimensional features, 4 heads,
and a Batch size of 1. Each model is run for ten for
ward passes on Google Colab T4-GPU, and we report the
mean runtime with standard deviation, throughput, and peak
memory usage. NAC occupies an intermediate position
in runtime relative to several CT-RNN models, including
GRU-ODE, CfC, and LTC. In terms of memory consumption, NAC uses significantly less memory than mTAN, with
NAC-2k being the least memory-consuming among the CTAttention models. Reducing NAC sparsity from 90% to 20%
has minimal effect on memory, decreasing usage slightly



7


**Neuronal Attention Circuit (NAC) for Representation Learning**



_Table 2._ Run-Time and Memory Benchmark Results


**Run-Time** **Throughput** **Peak Memory**
**Model**
(s) (seq/s) (MB)


RNN 1 _._ 8392 _[±]_ [0] _[.]_ [1933] 0.544 0.29
CT-RNN 7 _._ 1097 _[±]_ [0] _[.]_ [3048] 0.141 0.67
LSTM 2 _._ 6241 _[±]_ [0] _[.]_ [2906] 0.381 0.42
PhasedLSTM 4 _._ 9812 _[±]_ [0] _[.]_ [272] 0.201 0.80
GRU 3 _._ 216 _[±]_ [0] _[.]_ [2566] 0.311 0.54
GRU-ODE 12 _._ 2498 _[±]_ [0] _[.]_ [0525] 0.082 0.64
mmRNN 7 _._ 5852 _[±]_ [0] _[.]_ [2785] 0.132 0.96
LTC 14 _._ 643 _[±]_ [0] _[.]_ [2445] 0.068 0.99
CfC 6 _._ 0988 _[±]_ [0] _[.]_ [2135] 0.164 0.76


Attention 0 _._ 0016 _[±]_ [0] _[.]_ [0001] 625.00 16.86
MHA 0 _._ 0041 _[±]_ [0] _[.]_ [0001] 243.90 69.05
mTAN 0 _._ 0272 _[±]_ [0] _[.]_ [0054] 36.76 790.16
ODEFormer 0 _._ 0317 _[±]_ [0] _[.]_ [0016] 31.55 67.71
CTA 8 _._ 5275 _[±]_ [0] _[.]_ [2355] 0.117 1.43
ContiFormer 0 _._ 066 _[±]_ [0] _[.]_ [0075] 15.15 67.71


NAC-2k 7 _._ 3071 _[±]_ [0] _[.]_ [1547] 0.137 44.75
NAC-32k 7 _._ 2313 _[±]_ [0] _[.]_ [219] 0.138 549.86
NAC-PW 8 _._ 5649 _[±]_ [0] _[.]_ [0203] 0.117 5042.09
NAC-FC 0 _._ 0195 ~~_[±]_~~ ~~[0]~~ ~~_[.]_~~ ~~[0002]~~ 51.28 29.92
NAC-02s 7 _._ 252 _[±]_ [0] _[.]_ [2018] 0.138 151.54
NAC-09s 7 _._ 222 _[±]_ [0] _[.]_ [176] 0.139 150.85
NAC-Exact/05s/8k 7 _._ 4101 ~~_[±]_~~ ~~[0]~~ ~~_[.]_~~ ~~[1586]~~ 0.135 151.50
NAC-Euler 7 _._ 3367 _[±]_ [0] _[.]_ [1719] 0.136 152.22
NAC-Steady 7 _._ 2942 _[±]_ [0] _[.]_ [1451] 0.137 150.86


from 151.54 MB to 150.85 MB. In constrast, decreasing
the Top- _K_ selection from _PW_ to _k_ = 2 drastically reduces
memory consumption from 5042 MB to 44.75 MB, demonstrating the flexibility of NAC.
**Interpreting the Results:** From the experiments, we observe that increasing the sparsity of the NAC layer improves
the robustness of the system and leads to higher overall accuracy. Similarly, increasing the Top- _K_ interactions enhances
accuracy too; however, the benefits diminish as memory
consumption grows. Using Exact mode, Top- _K_ =8 with 50%
sparsity achieves the best balance between accuracy and
efficiency. Steady mode is the fastest, while Euler mode
handles adaptive temporal dynamics.


**4. Discussions**



This research is part of ongoing work on biologically plausible attention mechanisms and represents a pioneering step,
with limitations to be addressed in future work.

**Architectural improvement:** Currently, NAC uses predetermined wiring (AutoNCP) requiring three inputs: number of units (sensory + interneuron + motor), output motor
neurons, and sparsity, with typically 60% of units assigned
to sensory neurons. To integrate with the attention mechanism while preserving wiring, sensory units for _NN_ sensory
are set as unitssensory = � _d_ model0 _.−_ 6 0 _._ 5 � and backbone units as



are set as unitssensory = � _d_ model0 _.−_ 6 0 _._ 5 � and backbone units as

units _backbone_ = _d_ model + � _d_ 0model _._ 6 �, where _⌈·⌉_ and _⌊·⌋_ denote



units _backbone_ = _d_ model + � _d_ 0model _._ 6 �, where _⌈·⌉_ and _⌊·⌋_ denote

the ceiling and floor functions, respectively. This results
in a larger overall architectural size and increased runtime.



Future work will support user-defined NCPs configurations
or randomized wiring to enable more efficient architectures.
**Learnable sparse Top-** _**K**_ **selection:** Sparse Top- _K_ attention can miss important context, is sensitive to _k_, and may
be harder to optimize. A further limitation is that it still
computes the full _QK_ _[⊤]_ matrix, which can dominate the
cost for very long sequences. Future work includes adaptive
or learnable Top- _K_ selection, improved key scoring, and
hardware-aware optimization to strengthen accuracy and
robustness.


**5. Conclusion**


In this paper, we introduce the Neuronal Attention Circuit
(NAC), a biologically inspired attention mechanism that
reformulates attention logits as the solution to a first-order
ODE modulated by nonlinear, interlinked gates derived from
repurposing _C.elegans_ nematode NCPs. NAC bridges discrete attention with continuous-time dynamics, enabling
adaptive temporal processing without the limitations inherent to traditional scaled dot-product attention. Based on the
solution to ODE, we introduce three computational modes:
(i) Euler, based on _explicit Euler_ integration; (ii) Exact, providing closed-form solutions; and (iii) Steady, approximating equilibrium states. In addition, a sparse Top- _K_ pairwise
concatenation scheme is introduced to mitigate the memory intensity. Theoretically, we establish NAC’s log-state
stability, exponential error bounds, and universal approximation, thereby providing rigorous guarantees of convergence
and expressiveness. Empirical evaluations demonstrate that
NAC achieves state-of-the-art performance across diverse
tasks, including irregularly sampled time-series benchmarks,
autonomous vehicle lane-keeping, and industrial prognostics. Moreover, NAC occupies an intermediate position
between CT-RNNs and CT-Attention, offering robust temporal modeling while requiring less runtime than CT-RNNs
and less memory than CT-Attention models.


**Reproducibility Statement**


The code for reproducibility is available at
[https://github.com/itxwaleedrazzaq/](https://github.com/itxwaleedrazzaq/neuronal_attention_circuit)

[neuronal_attention_circuit](https://github.com/itxwaleedrazzaq/neuronal_attention_circuit)


**Impact Statement**


The work addresses the growing field of continuous-time
attention and pioneers a biologically plausible mechanism.
It encourages research into sparse, adaptive networks that
resemble natural wiring. From a societal perspective, it
supports more robust AI in resource-limited settings, but it
also raises ethical concerns when applied to areas such as
surveillance or autonomous systems.



8


**Neuronal Attention Circuit (NAC) for Representation Learning**



**References**


Introduction to self-driving cars. URL
[https://www.udacity.com/course/](https://www.udacity.com/course/intro-to-self-driving-cars--nd113)
[intro-to-self-driving-cars--nd113.](https://www.udacity.com/course/intro-to-self-driving-cars--nd113)


Aguiar-Conraria, L. and Soares, M. J. The continuous
wavelet transform: Moving beyond uni-and bivariate analysis. _Journal of economic surveys_, 28(2):344–375, 2014.


Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. _arXiv preprint_
_arXiv:2004.05150_, 2020.


Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B.,
Flepp, B., Goyal, P., Jackel, L. D., Monfort, M., Muller,
U., Zhang, J., et al. End to end learning for self-driving
cars. _arXiv preprint arXiv:1604.07316_, 2016.


Brockman, G., Cheung, V., Pettersson, L., Schneider, J.,
Schulman, J., Tang, J., and Zaremba, W. Openai gym.
_arXiv preprint arXiv:1606.01540_, 2016.


Cao, Y., Li, S., Petzold, L., and Serban, R. Adjoint sensitivity analysis for differential-algebraic equations: The
adjoint dae system and its numerical solution. _SIAM_
_journal on scientific computing_, 24(3):1076–1089, 2003.


Chen, R. T., Rubanova, Y., Bettencourt, J., and Duvenaud,
D. K. Neural ordinary differential equations. _Advances_
_in neural information processing systems_, 31, 2018.


Chen, Y., Ren, K., Wang, Y., Fang, Y., Sun, W., and Li, D.
Contiformer: Continuous-time transformer for irregular
time series modeling. _Advances in Neural Information_
_Processing Systems_, 36:47143–47175, 2023.


Chien, J.-T. and Chen, Y.-H. Continuous-time attention for
sequential learning. In _Proceedings of the AAAI confer-_
_ence on artificial intelligence_, volume 35, pp. 7116–7124,
2021.


Cho, K., Van Merrienboer, B., Gulcehre, C., Bahdanau,¨
D., Bougares, F., Schwenk, H., and Bengio, Y. Learning phrase representations using rnn encoder-decoder
for statistical machine translation. _arXiv preprint_
_arXiv:1406.1078_, 2014.


d’Ascoli, S., Becker, S., Mathis, A., Schwaller, P., and
Kilbertus, N. Odeformer: Symbolic regression of dynamical systems with transformers. _arXiv preprint_
_arXiv:2310.05573_, 2023.


De Brouwer, E., Simm, J., Arany, A., and Moreau, Y.
Gru-ode-bayes: Continuous modeling of sporadicallyobserved time series. _Advances in neural information_
_processing systems_, 32, 2019.



Deng, L. The mnist database of handwritten digit images
for machine learning research [best of the web]. _IEEE_
_signal processing magazine_, 29(6):141–142, 2012.


Ding, Y., Jia, M., Miao, Q., and Huang, P. Remaining
useful life estimation using deep metric transfer learning
for kernel regression. _Reliability Engineering & System_
_Safety_, 212:107583, 2021.


Hasani, R., Lechner, M., Amini, A., Rus, D., and Grosu,
R. Liquid time-constant networks. In _Proceedings of the_
_AAAI Conference on Artificial Intelligence_, volume 35,
pp. 7657–7666, 2021.


Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray,
A., Tschaikowski, M., Teschl, G., and Rus, D. Closed
form continuous-time neural networks. _Nature Machine_

_Intelligence_, 4(11):992–1003, 2022.


Hochreiter, S. The vanishing gradient problem during learning recurrent neural nets and problem solutions. _Interna-_
_tional Journal of Uncertainty, Fuzziness and Knowledge-_
_Based Systems_, 6(02):107–116, 1998.


Hochreiter, S. and Schmidhuber, J. Long short-term memory.
_Neural computation_, 9(8):1735–1780, 1997.


Hong, H. S. and Thuan, N. Hust bearing: a practical dataset
for ball bearing fault diagnosis. _Mendeley Data_, 3, 2023.


John, F. On integration of parabolic equations by difference methods: I. linear and quasi-linear equations for the
infinite interval. _Communications on Pure and Applied_
_Mathematics_, 5(2):155–211, 1952.


Jordan, M. I. Serial order: A parallel distributed processing
approach. In _Advances in psychology_, volume 121, pp.
471–495. Elsevier, 1997.


Lechner, M. and Hasani, R. Mixed-memory rnns for learning long-term dependencies in irregularly sampled time
series. 2022.


Lechner, M., Hasani, R. M., and Grosu, R. Neuronal circuit
policies. _arXiv preprint arXiv:1803.08554_, 2018.


Lechner, M., Hasani, R., Amini, A., Henzinger, T. A., Rus,
D., and Grosu, R. Neural circuit policies enabling auditable autonomy. _Nature Machine Intelligence_, 2(10):
642–652, 2020.


LeCun, Y., Touresky, D., Hinton, G., and Sejnowski, T. A
theoretical framework for back-propagation. In _Proceed-_
_ings of the 1988 connectionist models summer school_,
volume 1, pp. 21–28, 1988.


Lin, J. and Qu, L. Feature extraction based on morlet
wavelet and its application for mechanical fault diagnosis.
_Journal of sound and vibration_, 234(1):135–148, 2000.



9


**Neuronal Attention Circuit (NAC) for Representation Learning**



Nectoux, P., Gouriveau, R., Medjaher, K., Ramasso, E.,
Chebel-Morello, B., Zerhouni, N., and Varnier, C. Pronostia: An experimental platform for bearings accelerated
degradation tests. In _IEEE International Conference on_
_Prognostics and Health Management, PHM’12._, pp. 1–8.
IEEE Catalog Number: CPF12PHM-CDR, 2012.


Neil, D., Pfeiffer, M., and Liu, S.-C. Phased lstm: Accelerating recurrent network training for long or event-based
sequences. _Advances in neural information processing_
_systems_, 29, 2016.


Nishijima, T. Universal approximation theorem for neural
networks. _arXiv preprint arXiv:2102.10993_, 2021.


Park, M., Kim, H., and Park, S. A convolutional neural
network-based end-to-end self-driving using lidar and
camera fusion: Analysis perspectives in a real-world environment. _Electronics_, 10(21):2608, 2021.


Razzaq, W. and Hongwei, M. Neural circuit policies imposing visual perceptual autonomy. _Neural Processing_
_Letters_, 55(7):9101–9116, 2023.


Razzaq, W. and Zhao, Y.-B. Carle: a hybrid deep-shallow
learning framework for robust and explainable rul estimation of rolling element bearings. _Soft Computing_, 29(23):
6269–6292, 2025a.


Razzaq, W. and Zhao, Y.-B. Developing distance-aware uncertainty quantification methods in physics-guided neural
networks for reliable bearing health prediction, 2025b.
[URL https://arxiv.org/abs/2512.08499.](https://arxiv.org/abs/2512.08499)


Roy, A., Saffar, M., Vaswani, A., and Grangier, D. Efficient
content-based sparse attention with routing transformers. _Transactions of the Association for Computational_
_Linguistics_, 9:53–68, 2021.


Rubanova, Y., Chen, R. T., and Duvenaud, D. K. Latent
ordinary differential equations for irregularly-sampled
time series. _Advances in neural information processing_
_systems_, 32, 2019.


Rumelhart, D. E., Hinton, G. E., and Williams, R. J. Learning internal representations by error propagation. Technical report, 1985.


Shibuya, N. Car behavioral cloning, 2017. URL
[https://github.com/naokishibuya/](https://github.com/naokishibuya/car-behavioral-cloning)
[car-behavioral-cloning.](https://github.com/naokishibuya/car-behavioral-cloning) Accessed: 202510-05.


Shukla, S. N. and Marlin, B. M. Multi-time attention networks for irregularly sampled time series. _arXiv preprint_
_arXiv:2101.10318_, 2021.



Stinchcomb, M. Multilayered feedforward networks are
universal approximators. _Neural Networks_, 2:356–359,
1989.


Tay, Y., Bahri, D., Yang, L., Metzler, D., and Juan, D.-C.
Sparse sinkhorn attention. In _International conference on_
_machine learning_, pp. 9438–9447. PMLR, 2020.


Thuan, N. D. and Hong, H. S. Hust bearing: a practical
dataset for ball bearing fault diagnosis. _BMC research_
_notes_, 16(1):138, 2023.


Tiang, Y., Gelernter, J. R., Wang, X., Chen, W., Gao, J.,
Zhang, Y., and Li, X. Lane marking detection via fast
end-to-end deep convolutional neural network that is our
patch proposal network (ppn). 2018.


Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information_
_processing systems_, 30, 2017.


Vidulin, V., Lustrek, M., Kaluza, B., Piltaver, R.,
and Krivec, J. Localization Data for Person Activity. UCI Machine Learning Repository, 2010. DOI:
https://doi.org/10.24432/C57G8X.


Wang, B., Lei, Y., Li, N., et al. Xjtu-sy bearing datasets.
_GitHub, GitHub Repository_, 2018.


Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q.,
Yang, L., et al. Big bird: Transformers for longer sequences. _Advances in neural information processing_
_systems_, 33:17283–17297, 2020.


Zhuang, J., Dvornek, N., Li, X., Tatikonda, S., Papademetris,
X., and Duncan, J. Adaptive checkpoint adjoint method
for gradient estimation in neural ode. In _International_
_Conference on Machine Learning_, pp. 11639–11649.
PMLR, 2020.


Zhuang, J., Jia, M., Ding, Y., and Ding, P. Temporal
convolution-based transferable cross-domain adaptation
approach for remaining useful life estimation under variable failure behaviors. _Reliability Engineering & System_
_Safety_, 216:107946, 2021.



10


**Neuronal Attention Circuit (NAC) for Representation Learning**



**Appendix**


**A. Preliminaries**


**A.1. Attention Mechanism**


Attention mechanisms have become a cornerstone in

modern neural architectures, enabling models to dynamically focus on relevant parts of the input. The concept
was first introduced in the context of neural machine
translation, where it allowed the decoder to weight encoder
outputs according to their importance for generating each
target token. Formally, given a query vector _q ∈_ R _[d]_, key
vectors _K_ = [ _k_ 1 _, k_ 2 _, . . ., kn_ ] _∈_ R _[n][×][d]_, and value vectors
_V_ = [ _v_ 1 _, v_ 2 _, . . ., vn_ ] _∈_ R _[n][×][d]_, the attention mechanism can
be expressed in two steps:


1. Compute the scaled dot attention logits:


_ai_ = _[q][T][ k][i]_ (21)
~~_√_~~ _d_


2. Normalize the logits to get attention weights and compute the output:


_e_ _[a][i]_
_αi_ = softmax( _ai_ ) = ~~_n_~~ (22)
~~�~~ _j_ =1 _[e][a][j]_



exhibit a resting potential at _−_ 70 mV and an activation potential near _−_ 20 mV. Similarly, each _Nm_ is composed of
two subneurons, _Mp_ and _Mn_, and is driven by a controllable variable _y_, which also maps to a biologically plausible
range [ _−_ 70 mV _, −_ 20 mV]. The connections in the NCP
architecture are designed to reflect the biological sparsity
and abstraction of neural circuits. Specifically, connections
from _Ns_ to _Ni_ are feedforward, while those between _Nc_
and _Nm_ are highly recurrent (Lechner et al., 2018). Figure
1(a) illustrates the connectome of NCPs.


**B. Proofs**


In this section, we provide all the proofs.


**B.1. Deriving Closed-form (Exact) Solution**


Although _ϕ_ and _ωτ_ are nonlinear functions of the input
**u** = [ **q** ; **k** ], we derive closed-form solution by treating them
as locally constant over the pseudo-time integration interval
for each query–key pair based on frozen-coefficient approximation (John, 1952). This is accurate whenever the interval
is short or when input variations are slow compared with
the relaxation rate _ωτ_ . Under approximation assumption,
rewrite Eqn. 1 as


_dadtt_ [+] _[ ω][τ]_ _[a][t]_ [ =] _[ ϕ.]_ (24)


This is now a linear first-order ODE. The integrating factor
is



Attention( _q, k, v_ ) =



_n_
� _αivi_ (23)


_i_ =1



Here, _ai_ is the raw attention logit between the query and
each key, and the scaling factor _√d_ prevents large dot prod
ucts from destabilizing the softmax (Vaswani et al., 2017).


**A.2. Neuronal Circuit Policies (NCPs)**


NCPs represent a biologically inspired framework for developing interpretable neural control agents by adapting the
tap-withdrawal circuit found in the nematode _C. elegans_
(Lechner et al., 2018). Unlike traditional spiking neural
networks, the majority of neurons in this circuit exhibit
electronic dynamics, characterized by the passive flow of
electrical charges, resulting in graded potentials. NCPs
are structured as a four-layer hierarchical architecture comprising sensory neurons ( _Ns_ ), interneurons ( _Ni_ ), command
neurons ( _Nc_ ), and motor neurons ( _Nm_ ). The _Ns_ perceive
and respond to external stimulus inputs and are responsible for the initial signal transduction. Each _Ns_ consists
of subneurons _Sp_ and _Sn_ and a system variable _x_ . The
activation of _Sp_ and _Sn_ depends upon the sign of _x_ : _Sp_
becomes activated for _x >_ 0, whereas _Sn_ becomes activated for _x <_ 0. The variable _x_ is mapped to the membrane
potential range of [ _−_ 70 mV _, −_ 20 mV], which is consistent
with the biophysical behavior of nerve cells, which typically



� _ωτ dt_

_µ_ = _e_ � �


Multiply both sides by _µ_ ( _t_ ):



Substitute back:


_e_ _[ω][τ][ t]_ _at −_ _a_ 0 = _ϕ ·_ _[e][ω][τ][ t][ −]_ [1] _._ (30)

_ωτ_



= _e_ _[ω][τ][ t]_ _._ (25)




_[da][t]_
_e_ _[ω][τ][ t]_ _dt_ [+] _[ ω][τ]_ _[e][ω][τ][ t][a][t]_ [ =] _[ ϕe][ω][τ][ t][.]_ (26)


Recognize the left-hand side as the derivative of _e_ _[ω][τ][ t]_ _at_ :


_d_
� _e_ _[ω][τ][ t]_ _at_ � = _ϕe_ _[ω][τ][ t]_ _._ (27)
_dt_


Integrate from 0 to _t_ :


_t_
_e_ _[ω][τ][ t]_ _at −_ _e_ [0] _a_ 0 = _ϕ_ _e_ _[ω][τ][ s]_ _ds._ (28)
�0


Compute the integral (since _ωτ ̸_ = 0):



�0 _t_



_e_ _[ω][τ][ s]_ _ds_ = [1]
0 _ωτ_



_ωτ_



� _e_ _[ω][τ][ t]_ _−_ 1� _._ (29)



11


Rearrange:



**Neuronal Attention Circuit (NAC) for Representation Learning**


CASE 2: MULTIPLE CONNECTIONS ( _M >_ 1).



_e_ _[ω][τ][ t]_ _at_ = _a_ 0 + _[ϕ]_

_ωτ_


Divide both sides by _e_ _[ω][τ][ t]_ :


_at_ = _a_ 0 _e_ _[−][ω][τ][ t]_ + _[ϕ]_

_ωτ_



� _e_ _[ω][τ][ t]_ _−_ 1� _._ (31)


�1 _−_ _e_ _[−][ω][τ][ t]_ [�] _._ (32)



The ODE is



with per-connection equilibria _Aj_ = _ϕj/fj_ . The effective
equilibrium is



_da_

_dt_ [=] _[ −]_ � � _[M]_



_da_



� _fj_ � _a_ +

_j_ =1



_M_
� _fjAj,_ (37)

_j_ =1



Set _a_ _[∗]_ := _[ϕ]_ . Then _at_ = _a_ _[∗]_ + ( _a_ 0 _−_ _a_ _[∗]_ ) _e_ _[−][ω][τ][ t]_, proved.

_ωτ_


**B.2. Proof of Theorem 1**


We divide the proof into two parts: (i) the single-connection
case _M_ = 1, and (ii) the general multi-connection case
_M >_ 1. The main technique is to evaluate the ODE at
boundary values of the proposed invariant interval and show
that the derivative points inward, ensuring that trajectories

cannot escape.


CASE 1: SINGLE CONNECTION ( _M_ = 1).


The ODE reduces to



_A_ =



� _Mj_ =1 _[f][j][A][j]_
~~�~~ _Mj_ =1 _[f][j]_ _._ (38)



Since the weights ~~�~~ _fjfj_ are positive and sum to 1, _A_ is a
convex combination of _{Aj}_ . Therefore,


_A ∈_ [ _A_ [min] _, A_ [max] ] _._ (39)


- _Upper bound:_ Let _M_ = max(0 _, A_ [max] ). Then



_da_

_dt_ ��� _a_ = _M_ [=]



_M_
� _fj_ ( _Aj −_ _M_ ) _._ (40)

_j_ =1



_da_

_A_ = _[ϕ]_
_dt_ [=] _[ −][ω][τ]_ _[a]_ [ +] _[ ϕ]_ [ =] _[ −][ω][τ]_ [(] _[a][ −]_ _[A]_ [)] _[,]_ _ω_



_da_



_._ (33)
_ωτ_



Since _Aj ≤_ _A_ [max] _≤_ _M_, each term ( _Aj_ _−M_ ) _≤_ 0, and thus
� _fj_ ( _Aj −_ _M_ ) _≤_ 0. Hence _dadt_ _[≤]_ [0][, proving trajectories]

cannot exceed _M_ .


- _Lower bound:_ Let _m_ = min(0 _, A_ [min] ). Then



Here _A_ is the unique equilibrium. We now check both
bounds.


- _Upper bound:_ Let _M_ = max(0 _, A_ ). At _a_ = _M_,



_da_

_dt_ ��� _a_ = _m_ [=]



_M_
� _fj_ ( _Aj −_ _m_ ) _._ (41)

_j_ =1



_da_

(34)

_dt_ ��� _a_ = _M_ [=] _[ −][ω][τ]_ [(] _[M][ −]_ _[A]_ [)] _[.]_



If _A ≥_ 0, then _M_ = _A_ and _[da]_



If _A ≥_ 0, then _M_ = _A_ and _dt_ [= 0][. If] _[ A <]_ [ 0][, then] _[ M]_ [ = 0][,]

so _[da]_ _dt_ [=] _[ −][ω][τ]_ [(0] _[ −]_ _[A]_ [) =] _[ ω][τ]_ _[A][ ≤]_ [0][ since] _[ A <]_ [ 0][. In both]

cases, _[da]_ _[≤]_ [0][. Thus trajectories cannot cross above] _[ M]_ [.]



_dt_ _[≤]_ [0][. Thus trajectories cannot cross above] _[ M]_ [.]



- _Lower bound:_ Let _m_ = min(0 _, A_ ). At _a_ = _m_,



_da_

(35)

_dt_ ��� _a_ = _m_ [=] _[ −][ω][τ]_ [(] _[m][ −]_ _[A]_ [)] _[.]_



Since _Aj ≥_ _A_ [min] _≥_ _m_, each ( _Aj −_ _m_ ) _≥_ 0, so [�] _fj_ ( _Aj −_
_m_ ) _≥_ 0. Hence _[da]_ _dt_ _[≥]_ [0][, proving trajectories cannot fall]

below _m_ . Thus, the interval [ _m, M_ ] is forward-invariant.

_Remark_ 1 _._ This result guarantees that the continuous-time
attention state converges within a well-defined interval dictated by the per-connection equilibria. In particular, for the
single-connection case ( _M_ = 1), the state trajectory converges monotonically toward the closed-form equilibrium
solution (Eqn. 18) without overshoot.


**B.3. Proof for Theorem 2**


The proof proceeds constructively by showing that the NAC
layer can emulate a single-hidden-layer feedforward neural
network with nonlinear activations, which is a universal
approximator under the Universal Approximation Theorem
(UAT). We assume self-attention on a single-token input
_x ∈_ R _[n]_ (setting sequence length _T_ = 1) and focus on
the steady mode for simplicity. Without loss of generality,
set _d_ model = _n_ + _m_ or adjust as needed for dimensionality.
Constructively, set NCP sparsity _s_ = 0 for full connectivity, ensuring the backbone˜ _NN_ backbone approximates any
_ϕ_ : R [2] _[d]_ _→_ [0 _,_ 1] with error _< δ_ via stacked layers. For



If _A ≤_ 0, then _m_ = _A_ and _[da]_ _dt_ [= 0][. If] _[ A >]_ [ 0][, then]

_m_ = 0, so _[da]_ _dt_ [=] _[ −][ω][τ]_ [(0] _[ −]_ _[A]_ [) =] _[ ω][τ]_ _[A][ ≥]_ [0][. In both cases,]

_dadt_ _[≥]_ [0][. Thus trajectories cannot cross below] _[ m]_ [. Therefore,]

the interval [ _m, M_ ] is forward-invariant.


To see this explicitly under Euler discretization with step
size ∆ _t >_ 0,


_a_ ( _t_ + ∆ _t_ ) = _at_ + ∆ _t ·_ _[da]_ (36)

_dt_ _[.]_



At _a_ = _M_, _[da]_ _dt_ _[≤]_ [0 =] _[⇒]_ _[a]_ [(] _[t]_ [ + ∆] _[t]_ [)] _[ ≤]_ _[M]_ [. At] _[ a]_ [ =] _[ m]_ [,]

_dadt_ _[≥]_ [0 =] _[⇒]_ _[a]_ [(] _[t]_ [ + ∆] _[t]_ [)] _[ ≥]_ _[m]_ [. By induction over steps,]

_at ∈_ [ _m, M_ ] for all _t ∈_ [0 _, T_ ].



12


**Neuronal Attention Circuit (NAC) for Representation Learning**



multi-head, scale _H_ proportionally to target complexity,
with output projection _Wo_ aggregating as in classical UAT
proofs (Stinchcomb, 1989).
**Input Projections:** The input _x_ is projected via NCPbased sensory projections to obtain query _q_ = _q_ proj( _x_ ), key
_k_ = _k_ proj( _x_ ), and value _v_ = _v_ proj( _x_ ), each in R _[d]_ [model] . For
emulation, set _q_ proj = _k_ proj = _In_ (identity on R _[n]_ ) and adjust


**Head Splitting and Sparse Top-** _**k**_ **Pairwise Computation:**
Split into _H_ heads, yielding _q_ [(] _[h]_ [)] _, k_ [(] _[h]_ [)] _∈_ R _[d]_ per head _h_,
where _d_ = _d_ model _/H_ . For _T_ = 1, compute sparse top- _k_
pairs, but since _T_ = 1, _K_ eff = 1, yielding concatenated
pair _u_ [(] _[h]_ [)] = [ _q_ [(] _[h]_ [)] ; _k_ [(] _[h]_ [)] ] _∈_ R [2] _[d]_ . Since _q_ [(] _[h]_ [)] = _k_ [(] _[h]_ [)], this is

[ _x_ [(] _[h]_ [)] ; _x_ [(] _[h]_ [)] ], but the NCP processes it generally.
**Computation of** _ϕ_ [(] _[h]_ [)] **and** _ωτ_ [(] _[h]_ [)] **:** The scalar _ϕ_ [(] _[h]_ [)] is computed via the NCP-based inter-to-motor projection on the
pair:
_ϕ_ [(] _[h]_ [)] = _σ_ ( _NN_ backbone( _u_ [(] _[h]_ [)] )) (42)


where _σ_ ( _z_ ) = (1 + _e_ _[−][z]_ ) _[−]_ [1] is the sigmoid. This NCP, with
sufficiently large units and low sparsity, approximates any
continuous scalar function _ϕ_ [˜] : R [2] _[d]_ _→_ [0 _,_ 1] to arbitrary precision on compact sets (by the UAT for multi-layer networks
(Stinchcomb, 1989)). Similarly, _ωτ_ [(] _[h]_ [)] is computed via:


_ωτ_ [(] _[h]_ [)] = softplus( _NN_ backbone( _u_ [(] _[h]_ [)] )) + _ε,_ _ε >_ 0 (43)


By setting weights to make _ωτ_ [(] _[h]_ [)] _≡_ 1 (constant), the steadymode logit simplifies to _a_ [(] _[h]_ [)] = _ϕ_ [(] _[h]_ [)] _/ωτ_ [(] _[h]_ [)] = _ϕ_ [(] _[h]_ [)] . Thus,
_a_ [(] _[h]_ [)] _≈_ _σ_ � _w_ [(] _[h]_ [)] _x_ + _b_ [(] _[h]_ [)][�] for chosen weights _w_ [(] _[h]_ [)] _, b_ [(] _[h]_ [)], emulating a sigmoid hidden unit.
**Attention Weights and output:** For _T_ = 1, the softmax
over one “key” yields _α_ [(] _[h]_ [)] = exp( _a_ [(] _[h]_ [)] ) _/_ exp( _a_ [(] _[h]_ [)] ) = 1.
The head output is _y_ [(] _[h]_ [)] = � _T_ _[α]_ [(] _[h]_ [)] _[v]_ [(] _[h]_ [)] _[dt]_ [. Set] _[ v]_ [proj][ such that]

_v_ [(] _[h]_ [)] = 1 (scalar), yielding _y_ [(] _[h]_ [)] _≈_ _σ_ � _w_ [(] _[h]_ [)] _x_ + _b_ [(] _[h]_ [)][�] . For
vector-valued _v_ [(] _[h]_ [)], more complex combinations are possible, but scalars suffice here.
**Output Projection:** Concatenate head outputs: _Y_ =

[ _y_ [(1)] ; _y_ [(2)] ; _. . ._ ; _y_ [(] _[H]_ [)] ] _∈_ R _[H]_ . Apply the final dense layer:


_g_ ( _x_ ) = ( _Y · Wo_ ) + _bo ∈_ R _[m]_ _._ (44)


With _y_ [(] _[h]_ [)] _≈_ _σ_ � _w_ [(] _[h]_ [)] _x_ + _b_ [(] _[h]_ [)][�], this matches a single-hiddenlayer network with _H_ units. By the UAT, for large _H_, such
networks approximate any continuous _f_ on compact _K_ to
accuracy _ϵ_, by choosing appropriate _w_ [(] _[h]_ [)] _, b_ [(] _[h]_ [)] _, Wo, bo_ .


**C. Training, Gradients and Complexity**


**C.1. Gradient Characterization**


We analyze the sensitivity of the dynamics with respect to
the underlying learnable parameters. Specifically, we compute closed-form derivatives of both the steady state and the
full trajectory _at_ with respect to the parameters _ϕ_ and _ωτ_ .
These expressions illuminate how gradients flow through



the system, and provide guidance for selecting parameterizations that avoid vanishing or exploding gradients.


C.1.1. TRAJECTORY SENSITIVITIES FOR CLOSED-FORM


FORMULATION


The trajectory is given by


_at_ = _a_ _[∗]_ + ( _a_ 0 _−_ _a_ _[∗]_ ) _e_ _[−][ω][τ][ t]_ _,_ (45)


which depends on ( _ϕ, ωτ_ ) both through the equilibrium _a_ _[∗]_

and the exponential term.
**Derivative with respect to** _ϕ_ **:** We obtain



_Interpretation_ : The gradient with respect to _ωτ_ contains
a transient term proportional to _te_ _[−][ω][τ][ t]_, which dominates
at intermediate times, and a steady-state contribution proportional to _−ϕ/ωτ_ [2][, which persists asymptotically. Thus,]
sensitivity to _ωτ_ is time-dependent, peaking before vanishing exponentially in the transient component.


**C.2. Gradient-Based Training**


Like Neural ODEs (Chen et al., 2018) and CT-RNNs
(Rubanova et al., 2019), NAC produces differentiable computational graphs and can be trained using gradient-based
optimization, such as the adjoint sensitivity method (Cao
et al., 2003) or backpropagation through time (BPTT) (LeCun et al., 1988). In this work, we use BPTT exclusively,
as the adjoint sensitivity method can introduce numerical
errors (Zhuang et al., 2020).


**C.3. Efficiency and Complexity**


Table 3 summarizes the computational complexity of different sequence models. For sequence prediction over length
_n_ with hidden dimension _k_, RNNs scale linearly, _O_ ( _nk_ ),
while Attention and NAC scale quadratically, _O_ ( _n_ [2] _k_ ).
ODE-based models, such as LNNs, incur an additional
multiplicative factor _S_ for the number of solver steps.
For single-time-step prediction, RNNs and LSTMs require
_O_ ( _k_ ), whereas Attention and NAC require _O_ ( _nk_ ) when
recomputing attention over the full sequence.



_∂at_

_∂ϕ_ [= 1] _[ −]_ _ω_ _[e]_ _τ_ _[−][ω][τ][ t]_



_∂at_



(46)
_ωτ_



_Interpretation_ : For large _ωτ_, the gradient with respect to _ϕ_
saturates quickly but shrinks to scale _O_ (1 _/ωτ_ ), potentially
slowing learning of _ϕ_ . Conversely, very small _ωτ_ leads
to large steady-state gradients, which may destabilize optimization.

**Derivative with respect to** _ωτ_ **:** Here, both the equilibrium
and the decay rate depend on _ωτ_, yielding



_∂at_
= _−_ _[ϕ]_
_∂ωτ_ _ωτ_ [2]



�1 _−_ _e_ _[−][ω][τ][ t]_ [�] _−_ ( _a_ 0 _−_ _a_ _[∗]_ ) _t e_ _[−][ω][τ][ t]_ _._ (47)



13


**Neuronal Attention Circuit (NAC) for Representation Learning**



_Table 3._ Sequence and time-step prediction complexity. _n_ is the
sequence length and _k_ is the hidden/model dimension.


**Model** **Sequence** **Time-step**
RNN _O_ ( _nk_ ) _O_ ( _k_ )
Attention _O_ ( _n_ [2] _k_ ) _O_ ( _nk_ )
LNN (ODEsolve) _O_ ( _nk · S_ ) _O_ ( _k · S_ )
NAC-Exact _O_ ( _n_ [2] _k_ ) _O_ ( _nk_ )
NAC-Euler _O_ ( _n_ [2] _k_ ) _O_ ( _nk_ )


**D. Evaluation**


**D.1. Related Works**


The brief description for related works is divided into four
subcategories.
**DT-RNNs:** RNN (Rumelhart et al., 1985) captures sequential dependencies in time-series data by updating a hidden state from the current observation and the previous
state. LSTM (Hochreiter & Schmidhuber, 1997) extends
RNNs with input, output, and forget gates, allowing the
network to maintain and update long-term memory, which
improves modeling of long-term dependencies in time-series
sequences. GRU (Cho et al., 2014) simplifies the LSTM
architecture by combining the forget and input gates into a
single update gate, allowing efficient modeling of long-term
dependencies in time-series sequences.
**CT-RNNs:** CT-RNN (Rubanova et al., 2019) model temporal dynamics using differential equations, enabling hidden
states to evolve continuously over time in response to inputs, which is particularly useful for irregularly sampled
time-series data. PhasedLSTM (Neil et al., 2016) introduces a time gate that updates hidden states according to
a rhythmic schedule, enabling efficient modeling of asynchronous or irregularly sampled time-series. GRU-ODE
(De Brouwer et al., 2019) extends the GRU to continuous
time, evolving hidden states via ODEs to handle sequences
with non-uniform time intervals. mmRNN (Lechner &
Hasani, 2022) combines short-term and long-term memory units to capture both fast-changing and slowly evolving
patterns in sequential data. LTC (Hasani et al., 2021) use
neurons with learnable, input-dependent time constants to
adapt the speed of dynamics and capture complex temporal
patterns in continuous-time data. CfC (Hasani et al., 2022)
approximate LTC dynamics analytically, providing efficient
continuous-time modeling without relying on numerical
ODE solvers.

**DT-Attentions:** Attention (Vaswani et al., 2017) computes
attention weights by measuring similarity between queries
and keys, scaling the results, and applying softmax to weigh
time-step contributions. Multi-Head Attention (Vaswani
et al., 2017) applies multiple parallel scaled dot-product
attention mechanisms, capturing different types of temporal
dependencies simultaneously for complex time-series modeling.



**CT-Attentions:** mTAN (Shukla & Marlin, 2021) learns
continuous-time embeddings and uses time-based attention
to interpolate irregular observations into a fixed-length representation for downstream encoder-decoder modeling. CTA
(Chien & Chen, 2021) generalizes discrete-time attention
to continuous-time by representing hidden states, context
vectors, and attention scores as functions whose dynamics are modeled via neural networks and integrated using
ODE solvers for irregular sequences. ODEFormer(d’Ascoli
et al., 2023) trains a sequence-to-sequence transformer on
synthetic trajectories to directly output a symbolic ODE
system from noisy, irregular time-series data. ContiFormer
(Chen et al., 2023) builds a continuous-time Transformer by
pairing ODE-defined latent trajectories with a time-aware
attention mechanism to model dynamic relationships in irregular time-series data.


**D.2. Ablations Details**


The brief descriptions of variants and ablation are also divided into four subcategories:
**Top-** _**K**_ **Ablations:** _NAC-2k_ uses Top- _K_ =2 to compute the
logits and _NAC-32k_ uses Top- _K_ =32. All variants use the
exact computation mode with 50% sparsity.
**Sparsity Ablations:** _NAC-02s_ uses 20% sparsity to compute the logits and _NAC-09s_ uses 90%. _NAC-PW_ employs
full pairwise (non-sparse) concatenation for input curation.
_NAC-FC_ replaces the sparse NCP gating mechanism with
a simple fully connected layer. All variants use the exact
computation mode with Top- _K_ =8.
**Modes variants:** _NAC-Euler_ computes attention logits using the explicit Euler integration method. _NAC-Steady_ derives attention logits from the steady-state solution of the
exact formulation. _NAC-Exact/05s/8k_ computes attention
logits using the closed-form exact solution. It also overlaps
with other ablations, so we combined it into a single one.
All modes use Top- _K_ =8, 50% sparsity and _δt_ =1.0. The
sensitivity of NAC to _δt_ is visualized in Figure 4


_Figure 4._ Effect of _δt_ on output of NAC.



14




```json
"img_neuronal_attention_circuits_13_0": {
    "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/neuronal_attention_circuits.pdf-13-0.png",
    "page": 13,
    "section": "References",
    "image_relevance": "high",
    "image_type": "plot",
    "semantic_role": "illustrates",
    "caption": "A line plot illustrating the sensitivity of Accuracy to the parameter δt. The x-axis represents δt, ranging from approximately 0.1 to 2.0, while the y-axis denotes Accuracy values, which fluctuate between roughly 0.82 and 0.87 across the observed range of δt.",
    "depicted_concepts": [
      "Sensitivity analysis",
      "Accuracy",
      "δt (delta t)",
      "Line plot"
    ],
    "confidence": "high"
}
```
**Neuronal Attention Circuit (NAC) for Representation Learning**



**D.3. Experimental Details**


D.3.1. EVENT-BASED MNIST


**Dataset Explanation and Curation:** The MNIST dataset,
introduced by (Deng, 2012), is a widely used benchmark for
computer vision and image classification tasks. It consists
of 70,000 grayscale images of handwritten digits (0–9), each
of size 28 _×_ 28 pixels, split into 60,000 training and 10,000
testing samples.
**Preprocessing:** We follow the preprocessing pipeline described in (Lechner & Hasani, 2022), which proceeds as
follows. First, a threshold is applied to convert the 8-bit
pixel values into binary values, with 128 as the threshold
on a scale from 0 (minimum intensity) to 255 (maximum
intensity). Second, each 28 _×_ 28 image is reshaped into a
one-dimensional time series of length 784. Third, the binary
time series is encoded in an event-based format, eliminating
consecutive occurrences of the same value; for example, the
sequence [1 _,_ 1 _,_ 1 _,_ 1] is transformed into (1 _, t_ = 4). This encoding introduces a temporal dimension and compresses the
sequences from 784 to an average of 53 time steps. Finally,
to facilitate efficient batching and training, each sequence
is padded to a fixed length of 256, and the time dimension
is normalized such that each symbol corresponds to one
unit of time. The resulting dataset defines a per-sequence
classification problem on irregularly sampled time series.
**Neural Network Architecture:** We develop an end-to-end
hybrid neural network by combining compact convolutional
layers with NAC or counterparts baselines for fair comparison. Detailed hyperparameters and architectural specifications are provided in Table 4.


D.3.2. PERSON ACTIVITY RECOGNITION (PAR)


**Dataset Explanation and Curation:** We used the
Localized Person Activity Recognition dataset provided by
UC Irvine (Vidulin et al., 2010). The dataset comprises
25 recordings of human participants performing different
physical activities. The eleven possible activities are
“walking,” “falling,” “lying down,” “lying,” “sitting down,”
“sitting,” “standing up from lying,” “on all fours,” “sitting
on the ground,” “standing up from sitting,” and “standing
up from sitting on the ground.” The objective of this
experiment is to recognize the participant’s activity from
inertial sensors, formulating the task as a per-time-step
classification problem. The input data consist of sensor
readings from four inertial measurement units placed on
participants’ arms and feet. While the sensors are sampled
at a fixed interval of 211 ms, recordings exhibit different
phase shifts and are thus treated as irregularly sampled time
series.

**Preprocessing:** We first separated each participant’s
recordings based on sequence identity and calculated
elapsed time in seconds using the sampling period. To



mitigate class imbalance, we removed excess samples from
overrepresented classes to match the size of the smallest
class. Subsequently, the data were normalized using a
standard scaler. Finally, the dataset was split into a 90:10
ratio for training and testing.
**Neural Network Architecture:** Following the approach in
Section D.3.1, we developed an end-to-end hybrid neural
network combining convolutional heads with NAC or other
baselines. Hyperparameter details are summarized in Table
4.


D.3.3. AUTONOMOUS VEHICLE


**Dataset Explanation and Curation:** We followed the data
collection methodology described in (Razzaq & Hongwei,
2023). For OpenAI-CarRacing, a PPO-trained agent (5M
timesteps) was used to record 20 episodes, yielding approximately 48,174 RGB images of size 92 _×_ 92 _×_ 3 with
corresponding action labels across five discrete actions (noact, move left, forward, move right, stop). The dataset was
split with 10% reserved for testing and the remaining 90%
for training. For the Udacity simulator, we manually controlled the vehicle for 50 minutes, producing 15647 RGB
images of size 320 _×_ 160 _×_ 3, captured from three camera
streams (left, center, right) along with their corresponding
continuous steering values. This dataset was split into 20%
testing and 80% training.
**Preprocessing:** No preprocessing was applied to the
OpenAI-CarRacing dataset. For the Udacity simulator, we
followed the preprocessing steps in (Shibuya, 2017). Each
image was first cropped to remove irrelevant regions and
resized to 66 _×_ 120 _×_ 3. Images were then converted from
RGB to YUV color space to match the network input. To improve robustness, data augmentation techniques, including
random flips, translations, shadow overlays, and brightness
variations, were applied to simulate lateral shifts and diverse
lighting conditions.
**Neural Network Architecture:** For OpenAI-CarRacing,
we modified the neural network architecture proposed in
(Razzaq & Hongwei, 2023), which combines compact CNN
layers for spatial feature extraction with LNNs to capture
temporal dynamics. In our implementation, the LNN layers
were replaced with NAC and its comparable alternatives
for fair evaluation. Full hyperparameter configurations are
provided in Table 4. For the Udacity simulator, we modified
the network proposed in (Bojarski et al., 2016) by replacing
three latent MLP layers with NAC and its counterparts. Full
hyperparameters for this configuration are summarized in
Table 4.

**Saliency Maps:** A saliency map visualizes the regions of
the input that a model attends to when making decisions.
Figure 5 shows the saliency maps for the OpenAI CarRacing environment. We observe that only NAC (Steady, Euler,



15


**Neuronal Attention Circuit (NAC) for Representation Learning**


_Figure 5._ Saliency maps for OpenAI CarRacing


_Figure 6._ Saliency maps for Udacity Simulator




```json
"img_neuronal_attention_circuits_15_0": {
    "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/neuronal_attention_circuits.pdf-15-0.png",
    "page": 15,
    "section": "References",
    "image_relevance": "high",
    "image_type": "attention_map",
    "semantic_role": "illustrates",
    "caption": "The image displays saliency maps generated for various recurrent neural network architectures applied to the OpenAI CarRacing environment. It includes traditional models like RNN, LSTM, and GRU, alongside continuous-time variants such as CT-RNN, PhasedLSTM, GRU-ODE, ODEFormer, ContiFormer, and NAC models. The top-left panel shows the input frame, while the subsequent panels visualize the regions of the input image that are most salient for each specified model.",
    "depicted_concepts": [
      "Saliency map",
      "OpenAI CarRacing",
      "Recurrent Neural Network",
      "RNN",
      "LSTM",
      "GRU",
      "Continuous-time Neural Network",
      "CT-RNN",
      "PhasedLSTM",
      "GRU-ODE",
      "mmRNN",
      "LTC",
      "CfC",
      "Attention mechanism",
      "MHA (Multi-Head Attention)",
      "mTAN",
      "CTA",
      "ODEFormer",
      "ContiFormer",
      "Neural Accumulator (NAC)",
      "NAC-Exact",
      "NAC-Euler",
      "NAC-Steady"
    ],
    "confidence": "high"
}
```


```json
"img_neuronal_attention_circuits_15_1": {
    "path": "E:/Python Stuff/MAS-for-multimodal-knowledge-graph/markdown_outputs/images/neuronal_attention_circuits.pdf-15-1.png",
    "page": 15,
    "section": "References",
    "image_relevance": "high",
    "image_type": "attention_map",
    "semantic_role": "illustrates",
    "caption": "A grid of saliency maps visualizes the regions of an input image from the Udacity Simulator that are most attended to by 16 different recurrent neural network architectures. Each map corresponds to a specific model, including RNN, LSTM, GRU, CT-RNN, PhasedLSTM, GRU-ODE, mmRNN, LTC, CfC, Attention, MHA, mTAN, CTA, ODEFormer, ContiFormer, NAC-Exact, NAC-Euler, and NAC-Steady models, showing varying patterns of attention across the road scene.",
    "depicted_concepts": [
      "Saliency map",
      "Udacity Simulator",
      "Recurrent Neural Network",
      "RNN",
      "LSTM",
      "GRU",
      "CT-RNN",
      "PhasedLSTM",
      "GRU-ODE",
      "mmRNN",
      "LTC",
      "CfC",
      "Attention mechanism",
      "MHA",
      "mTAN",
      "CTA",
      "ODEFormer",
      "ContiFormer",
      "NAC-Exact",
      "NAC-Euler",
      "NAC-Steady"
    ],
    "confidence": "high"
}
```

and Steady) maintains focus on the road’s horizon, while
other models either focus on the sides or remain largely
unresponsive to the task. Figure 6 also presents the saliency
maps for the Udacity Simulator. In this case, NAC-Exact,
CTA produces the most accurate visual maps, maintaining
attention on the road’s horizon, followed by ContiFormer
and mTAN, which achieve comparable performance. Attention and PhasedLSTM also generate reasonable saliency
maps, although their focus is more dispersed across the
scene. In contrast, other models either fail to identify relevant regions, producing blurry maps, or focus solely on
one side of the road. These results demonstrate the NAC’s

ability to understand the underlying task.


D.3.4. INDUSTRY 4.0


**Dataset Explanation and Curation:** _PRONOSTIA dataset_
is a widely recognized benchmark dataset in the field of
condition monitoring and degradation estimation of rollingelement bearings. Nectiux et al.(Nectoux et al., 2012) developed this dataset as part of the PRONOSTIA experimental
platform. The dataset comprises 16 complete run-to-failure
experiments performed under accelerated wear conditions



**Algorithm 3** Time–frequency Representation Algorithm


**Require:** windowed signal _Iw_, critical frequency _fc_, operating
frequency _fo_, sampling period _Tsampling_, windowed physical
constraints ( _tw, Tw_ )
_amin_ = _fmax·Tfsamplingc_ _[,]_ _amax_ = _f_ min _·Tsamplingfc_ [,]
_ascale ∈_ [ _amin, amax_ ]
_IT F R ←{}_
**for** ( _iw, tn, Tn_ ) in ( _Iw, tw, Tw_ ): **do**

Wavelets: Γ _iw_ ( _a, b_ ) = � _−∞∞_ _[i][w][ψ][∗]_ [�] _[t][−]_ _a_ _[b]_ � _dt._

Energy: _E_ = [�] _[M]_ _m_ =1 _[|]_ [Γ] _[iw]_ [(] _[a, b]_ [)] _[|]_ [2]

Dominant frequency: _fd_ = _a_ scale [arg max( _E_ )].
Entropy: _h_ = _−_ [�] _[M]_ _i_ = _m_ _[P]_ [(] _[i][w]_ [(] _[t]_ [)) log] _[ P]_ [(] _[i][w]_ [(] _[t]_ [))][.]

Kurtosis: _K_ = [E][[(] _[i][w]_ [(] _σ_ _[t]_ [4][)] _[−][µ]_ [)][4][]] .

Skewness: _sk_ = [E][[(] _[i][w]_ [(] _σ_ _[t]_ [3][)] _[−][µ]_ [)][3][]] .

1 _M_
mean: _µ_ = _M_ � _m_ =1 _[i][w]_ [(] _[m]_ [)][.]


1 _M_

standard deviation: _σ_ = ~~�~~ _M_ ~~�~~ _i_ =1 [(] _[i][w]_ [(] _[m]_ [)] _[ −]_ _[µ]_ [)][2][.]

_Xn ←_ [log( _E_ ) _, fd, h, K, sk, µ, σ_ ]
**end for**
**return** _IT F R_ = _Concat_ ( _X_ 1 _, X_ 2 _. . . XNs_ _, tn, Tn_ )



16


**Neuronal Attention Circuit (NAC) for Representation Learning**


_Table 4._ Summary of Key Hyperparameters of All Experiments


**Param.** **MNIST** **PAR** **CarRacing** **Udacity** **RUL EST.**


Conv layers 2× **1D** (64@5) **1D** (64@5, 64@3) 3×TD- **2D** (10–30@3–5) 5× **2D** (24–64@5–3, ELU) 2× **1D** (32@3, 16@2)
NAC 64-d, 8h 32-d, 4h 64-d, 16h 100-d, 16h 16-d, 8h
Dense 32–10(SM) 32–11(SM) 64–5(SM) 64–1(Lin) 1(Lin)
Dropout – – 0.2 0.5 –
Opt. AdamW AdamW Adam AdamW AdamW
LR 0.001 0.001 0.0001 0.001 0.001

Loss SCE SCE SCE MSE MSE

Metric Acc Acc Acc MAE Score

Batch 32 20 32 – 32

Epochs 150 500 100 10 150


**Note:** SCE = Sparse Categorical Crossentropy; Acc = Accuracy; MAE = Mean Absolute Error; MSE = Mean Squared Error; SM =
softmax; Lin = Linear; TD = TimeDistributed; Conv1D/2D = Conv1D/2D; _d_ = model dimension; _h_ = attention heads.

**Baselines Hyperparameters Clarification:** All (CT & DT) RNNs use the same number of hidden units as NAC’s _d_ model, and all (DT &
CT) Attention use the same _d_ model and _heads_ as NAC. The other layers, including 1D/2D, Dense, and the remaining hyperparameters, are
the same during our tests.


_Table 5._ Data distributions of PRONOSTIA, XJTU-SY, and HUST datasets.


**Dataset** **Condition** **Frequency** **Radial Load** **Speed** **Train** **Test**

_Condition 1_ 100 Hz 4 kN 1800 rpm 4 _∼_ 7 1 _∼_ 3
**PRONOSTIA** _Condition 2_ 100 Hz 4.2 kN 1650 rpm      - 1 _∼_ 7
_Condition 3_ 100 Hz 5 kN 1500 rpm                - 1 _∼_ 3

_Condition 1_ 35 Hz 4 kN 2100 rpm                - 1 _∼_ 5
**XJTU-SY** _Condition 2_ 37.5 Hz 4.2 kN 2250 rpm       - 1 _∼_ 5
_Condition 3_ 40 Hz 5 kN 2400 rpm                - 1 _∼_ 5

_Condition 1_                - 0 W                -                - 1 _∼_ 5

**HUST** _Condition 2_       - 200 W       -       - 1 _∼_ 5

_Condition 3_                - 400 W                -                - 1 _∼_ 5


**Note:** The PRONOSTIA dataset is utilized for training and generalization testing, while the XJTU-SY and HUST datasets are employed
to evaluate cross-validation testing. (-) values are either not available or not utilized.



across three different operating settings: 1800 rpm with 4
kN radial load, 1650 rpm with a 4.2 kN load, and 1500 rpm
with a 5 kN load, all at a frequency of 100 Hz. Vibration
data were recorded using accelerometers placed along the
horizontal and vertical axes, which were sampled at 25.6
kHz. Additionally, temperature readings were collected at a
sampling rate of 10 Hz. The data distributions for training
and testing are provided in Table 5.
_XJTU-SY Dataset_ is another widely recognized benchmark
dataset developed through collaboration between Xi’an
Jiaotong University and Changxing Sumyoung Technology
(Wang et al., 2018). The dataset comprises 15 complete runto-failure experiments performed under accelerated degradation conditions with three distinct operational settings: 1200
rpm (35 Hz) with a 12 kN radial load, 2250 rpm (37.5 Hz)
with an 11 kN radial load, and 2400 rpm (40 Hz) with a 10
kN radial load. Vibrational signals were recorded using an
accelerometer mounted on the horizontal and vertical axes

and sampled at 25.6 kHz. This dataset is only used for the



cross-validation test.

_HUST Dataset_ is a practical dataset developed by Hanoi
University of Science and Technology to support research
on ball bearing fault diagnosis (Hong & Thuan, 2023). The
dataset includes vibration data collected from five bearing
types (6204, 6205, 6206, 6207, and 6208) under three different load conditions: 0 W, 200 W, and 400 W. Six fault
categories were introduced, consisting of single faults (inner race, outer race, and ball) and compound faults (inner–outer, inner–ball, and outer–ball). Faults were created
as early-stage defects in the form of 0.2 mm micro-cracks,
simulating real degradation scenarios. The vibration signals
were sampled at 51.2 kHz with approximately 10-second
recordings for each case. This dataset is only used for the
cross-validation test.

**Preprocess:** Condition monitoring data comprises 1D nonstationary vibrational signals collected from multiple sensors. To extract meaningful information, these signals must
be transformed into features that possess meaningful physi


17


**Neuronal Attention Circuit (NAC) for Representation Learning**


cal interpretability. We utilized the preprocessing proposed
in (Razzaq & Zhao, 2025a) and labels are generated according to (Razzaq & Zhao, 2025b). Initially, the signal
is segmented into small, rectangularized vectors using a
windowing technique ( _w_ ), enabling better localization of
transient characteristics. The continuous wavelet transform

(CWT) (Aguiar-Conraria & Soares, 2014) with the Morlet
wavelet (Lin & Qu, 2000) as the mother wavelet is then applied to obtain a time-frequency representation (TFR). The
CWT is defined as Γ( _a, b_ ) = � _−∞∞_ _[x][w]_ [(] _[t]_ [)] ~~_√_~~ [1] ~~_a_~~ _ψ_ _[∗]_ [�] _[t][−]_ _a_ _[b]_ � _dt_,

where _a_ and _b_ denote the scale and translation parameters,
respectively, and _ψ_ is the Morlet wavelet function. From
the resulting TFR, a compact set of statistical and domainspecific features is extracted to characterize the operational
condition of the bearing. The complete feature extraction
procedure is described in Algorithm 3.
**Neural Network Architecture:** The objective of this problem is to design a compact neural network that can effectively model degradation dynamics while remaining feasible
for deployment on resource-constrained devices, enabling
localized and personalized prognostics for individual machines. To achieve this, we combine a compact convolutional network with NAC. The CNN component extracts
spatial degradation features from the training data, while
NAC performs temporal filtering to emphasize informative
features. This architecture maintains a small model size

without sacrificing representational capacity. Full hyperparameter configurations are reported in Table 4.
**Evaluation Metric:** Score is a metric specifically designed
for RUL estimation in the IEEE PHM (Nectoux et al., 2012)
to score the estimates. The scoring function is asymmetric and penalizes overestimations more heavily than early
predictions. This reflects practical considerations, as late
maintenance prediction can lead to unexpected failures with
more severe consequences than early intervention can.



_yi_ ˆ _−yi_

10 _−_ 1
�


(48)



_Score_ = �

_i_ :ˆ _yi<yi_



_e_ _[−]_ _[yi]_ [ˆ] 13 _[−][yi]_ _−_ 1 +
� � �

_i_ :ˆ _yi≥yi_



_e_
�



18


