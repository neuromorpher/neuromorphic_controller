# Spiking Neurons from Scratch — Chapter 2

### Eligibility Traces and Spike-Timing-Dependent Plasticity (STDP)

*The chapter where a weight finally learns. Built from your own work; every number here is one you produced and verified.*

---

## How to use this document

Same as Chapter 1: read it once for the narrative, then a second time with a pen, redoing every derivation before looking at the worked arithmetic. Stop at each **Checkpoint** and answer before continuing. Attempt the exam cold — the answer key is at the very bottom and reading it first destroys the point.

The one-line thesis of this chapter: **a synapse cannot store spike times, so it stores a fading trace instead — and that trace secretly computes the spike-timing math for free.**

---

## Learning objectives

By the end you should be able to, from memory:

1. Explain what an eligibility trace is and what single question it answers.
2. Derive the exponential decay multiplier `e^(-dt/tau)` from the idea of "keep a constant fraction each step."
3. Explain why the trace value at the moment of a spike equals `e^(-Δt/τ)` — i.e. why the trace *is* the timing computation.
4. State the two-branch STDP rule, including which spike reads which trace and the sign of each.
5. Explain the causal logic: why pre-before-post strengthens and post-before-pre weakens.
6. Predict the direction and rough magnitude of a weight change given a spike-timing protocol.
7. Explain why STDP alone is insufficient for a goal-directed agent (the motivation for Chapter 3).

---

## 1. Where Chapter 1 left a gap

At the end of Chapter 1 you had two neurons connected by a synapse with a **fixed** weight. The network could propagate activity but could not *learn* — nothing ever changed the weight. This chapter makes the weight change based on experience, using only information physically available at the synapse.

Recall the punchline of Chapter 1's final section: a synapse has access to exactly two timestamps — when its pre neuron fired and when its post neuron fired — and the quantity that matters is their difference, Δt = t_post − t_pre. The sign of Δt infers causal direction; its magnitude infers confidence. STDP is the rule that turns Δt into a weight change. The only remaining problem is *mechanism*: how does a synapse compute anything about Δt when it can't store timestamps?

---

## 2. The eligibility trace

### 2.1 What it is

Give each neuron a single number — its **trace** — that does exactly one thing:

- it **jumps to 1.0** the instant the neuron fires, and
- it **decays exponentially toward 0** on every timestep where the neuron is silent.

That's the entire object. It is a **recency meter**. It answers one question at any moment: *how recently did this neuron fire?*

- trace ≈ 1.0 → fired just now
- trace ≈ 0.37 → fired about one time constant ago
- trace ≈ 0.0 → hasn't fired in a long time

A trace only ever jumps for **its own** neuron's spike. It does not respond to any other neuron.

### 2.2 Where the trace lives, physically

The trace is a modeling abstraction, but it stands in for real physical residue that a spike leaves behind — chiefly **calcium** flooding the cell and synaptic terminals and then being pumped out gradually over tens of milliseconds, plus lingering neurotransmitter/receptor activation (e.g. NMDA receptor states) and residual dendritic depolarization from the backpropagating spike. Modeling every ion would be intractable and teach nothing extra, so we compress all of it into one number: "how much chemical evidence exists that this neuron just fired."

This is not just convenience. That fading calcium is *literally how a real synapse solves the timing problem.* A synapse can't read a clock, but it can "feel" how much calcium is currently present. If pre fired and left calcium, and post then fires while that calcium is still high, the synapse sits in a chemical state that physically encodes "these two fired close together." That coincidence is what triggers strengthening. The trace is the mechanism nature uses to convert "I only know the present instant" into "I can detect that two events happened near in time."

> **Checkpoint 2.A.** In one sentence, what question does the trace answer? And: if a neuron fires while its trace is already at 0.6, what does the trace become?

---

## 3. The decay term, derived

The line `self.trace *= np.exp(-dt / self.tau_trace)` is the heart of the mechanism. Do not treat it as a black box.

### 3.1 Decay is repeated multiplication by a fraction below 1

Suppose each millisecond a value keeps 90% of itself, starting at 1.0: 1.0 → 0.9 → 0.81 → 0.729 → … That is all decay is — multiply by a constant fraction less than 1 every step; the value approaches 0 without ever jumping. The whole question is *what fraction.*

### 3.2 What should control the fraction

Two things:

- **dt (timestep size).** More time elapsed per step → more shrink. So dt belongs on top, increasing the decay.
- **tau (time constant).** Bigger tau → *slower* decay, longer memory (same meaning as in the neuron: big tau = leaks slowly). So tau belongs on the bottom.

The natural quantity is the ratio **dt / tau** — "how big is this step relative to the memory timescale." Small ratio → barely fades; large ratio → fades a lot.

### 3.3 Why the exponential

We want smooth, continuous decay that gives the *same* answer at a given time regardless of how finely we chop dt. The function whose rate of decrease is proportional to its current value — i.e. "lose a constant fraction per instant" taken to the continuous limit — is the exponential. So the per-step multiplier is:

    e^(-dt / tau)

With dt = 0.5, tau = 20: e^(-0.5/20) = e^(-0.025) ≈ 0.975. Each 0.5 ms step keeps 97.5% of the trace.

### 3.4 The number to memorize

After one full time constant tau, you have multiplied by e^(-1) ≈ **0.368**. So after tau milliseconds, *any* exponentially-decaying quantity has fallen to ~37% of its start. Always. After 2·tau: e^(-2) ≈ 0.135 (~13.5%).

This is the identical math as Chapter 1 Exam B4, where a neuron recovered as e^(-40/20) = e^(-2) ≈ 0.135. Same e^(-time/tau) shape — the trace just applies it every step instead of once.

> **Checkpoint 3.A.** With tau_trace = 20, a trace is at 1.0 right after a spike. Roughly what is it 20 ms later? 40 ms later? (Use the 37% rule; no calculator.)

---

## 4. The punchline: the trace *is* the timing computation

Here is why the trace is elegant rather than just bookkeeping.

Suppose pre fires. Pre's trace jumps to 1.0 and begins decaying. Then Δt milliseconds later, post fires. **At that moment, what is pre's trace?** It is exactly:

    pre.trace = e^(-Δt / tau_trace)

That is *the exponential-of-Δt the STDP rule wants* — computed automatically, with no timestamp ever stored. A large pre-trace at the moment post fires means "pre fired very recently" (small Δt → strong effect). A tiny pre-trace means "pre fired long ago" (large Δt → negligible effect). The decay did the math.

So reading the *other* neuron's trace at the instant of a spike hands you the timing-dependence for free. That is the whole trick.

> **Checkpoint 4.A.** Pre fires, then post fires 3 ms later, tau_trace = 20. What is pre's trace when post fires? (Compute e^(-3/20).) Is that a strong or weak weight change?

---

## 5. Two clocks: tau vs tau_trace

The neuron now carries two time constants, and they are **independent**:

- **tau (membrane)** — the *electrical* clock: how fast voltage relaxes toward rest, set by the membrane's resistance and capacitance.
- **tau_trace** — the *chemical* clock: how fast the spike's residue (calcium, receptor activation) clears.

They share the same decay *math* (e^(-dt/tau)) but represent different physical processes, so there is no reason they must be equal. Functionally, tau_trace sets the **width of the STDP learning window** — how close in time two spikes must be to count as related. Short tau_trace = narrow window (only tight coincidences matter); long tau_trace = wide window (looser associations). You will want to tune that independently of the neuron's firing behavior, which is why they are separate variables even though both default to 20.0.

(Caveat for honesty: the membrane clock is *primarily* electrical but its resistance is set by ion channels, which are chemical; and real traces carry some recency in residual voltage too. "Electrical clock vs chemical clock" is the right working distinction, not a perfectly hard wall.)

> **Checkpoint 5.A.** If you wanted the neuron to only strengthen connections between spikes that occur *very* close together, would you make tau_trace larger or smaller? Why?

---

## 6. The STDP rule

### 6.1 The two branches

Each event reads the *other* neuron's trace:

- **When post fires** → read **pre's** trace → **strengthen** (`+=`):
  `weight += A_plus * pre.trace`
- **When pre fires** → read **post's** trace → **weaken** (`-=`):
  `weight -= A_minus * post.trace`

Both are independent `if` checks (not `elif`) — in principle both neurons can fire the same timestep, and both updates should apply.

### 6.2 The causal logic

- **post fires, pre fired recently** (pre.trace high): pre plausibly *caused* post → reward that connection → strengthen. Consistent with Δt > 0 (pre before post).
- **pre fires, post fired recently** (post.trace high): post already fired *before* pre, so pre arrived too late to have caused anything → prune that connection → weaken. Consistent with Δt < 0 (post before pre).

This "prune" is exactly the maladaptive-connection weakening the project brief referred to. The sign of the weight change tracks the *order* of firing, not merely whether spikes occurred.

### 6.3 The code

```python
class Synapse:
    def __init__(self, pre, post, weight, A_plus=0.01, A_minus=0.01):
        self.pre = pre
        self.post = post
        self.weight = weight
        self.A_plus = A_plus
        self.A_minus = A_minus

    def get_current(self, pre_spiked: bool):
        if pre_spiked:
            return self.weight
        else:
            return 0.0

    def update_weight(self, pre_spiked: bool, post_spiked: bool):
        if post_spiked:
            self.weight += self.A_plus * self.pre.trace
        if pre_spiked:
            self.weight -= self.A_minus * self.post.trace
```

And the trace additions to `LIFNeuron`: a `tau_trace` parameter and `self.trace = 0.0` in `__init__`; in `step`, decay the trace **first** every timestep (`self.trace *= np.exp(-dt/self.tau_trace)`), then set `self.trace = 1.0` **inside** the spike block. Decay-first-then-bump guarantees the trace reads exactly 1.0 on the spike step and starts fading the next step.

> **Checkpoint 6.A.** At the very start of a simulation both traces are 0.0. The first time post fires, before pre has ever fired, how much does the weight change? Why?

---

## 7. Watching it learn

### 7.1 The causal protocol (using modular timing)

To force pre to fire ~3 ms before post, repeatedly, we use modular arithmetic. `t % P` is a sawtooth that ramps 0→(P−dt) then resets, every P ms — a clock telling you "how far into the current cycle are we." The recipe "pulse for width W starting at offset D, every period P" is:

    if (t % P) >= D and (t % P) < D + W:

with the rule that **W must be at least dt**, or the window can slip between timesteps and never fire.

```python
pre = LIFNeuron(); post = LIFNeuron()
syn = Synapse(pre, post, weight=5.0)   # start weak
dt = 0.5

for t in np.arange(0, 500, dt):
    I_pre  = 40.0 if (t % 50) < 1.0 else 0.0                      # pre at cycle start
    I_post = 40.0 if (t % 50) >= 3.0 and (t % 50) < 4.0 else 0.0  # post 3 ms later
    ps = pre.step(I_pre, t, dt)
    qs = post.step(I_post, t, dt)
    syn.update_weight(ps, qs)
```

### 7.2 The result, and the subtlety in the number

Verified output: weight went **5.0000 → 5.0775** over 10 cycles. It *increased* — the connection learned. But the rise is slightly *less* than the naive "10 × A_plus × 0.86 ≈ 0.086" estimate, and the gap is instructive.

Both neurons spike every cycle (pre at 0.5, 50.5, …; post at 3.5, 53.5, …), so **both** branches fire each cycle:

- **post fires 3 ms after pre** → reads pre.trace ≈ e^(-3/20) ≈ 0.86 → strengthen ≈ 0.01 × 0.86 = **+0.0086**
- **pre fires ~47 ms after the previous post** → reads post.trace ≈ e^(-47/20) ≈ 0.095 → weaken ≈ 0.01 × 0.095 = **−0.00095**

Net per cycle ≈ **+0.0077**; ten cycles ≈ **+0.077**. That matches 5.0775.

The asymmetry is the whole point: the strengthen term is large (pre fired *recently* before post) and the weaken term is tiny (post fired *long* before the next pre). Net positive because the ordering is causal.

### 7.3 Flipping the timing

Reverse the offsets so **post fires before pre**, and the two terms swap magnitudes: verified output **5.0000 → 4.9225**, a decrease of ~0.077 — equal in size, opposite in sign. This is the STDP curve being **antisymmetric** about Δt = 0: causal order strengthens, acausal order weakens, symmetrically.

> **Checkpoint 7.A.** In the causal run, why is the weaken term so much smaller than the strengthen term, given both use the same A = 0.01?

---

## 8. Common pitfalls (all seen this chapter)

1. **Neuron never spikes, so the trace stays 0.** A too-short current kick (e.g. current on for only 2 steps landing *exactly* on threshold) grazes rather than crosses. Predict the voltage arithmetic; give a clear, unambiguous kick. Empty `spike_times` is the tell.
2. **Bump before decay.** If you set trace = 1.0 and *then* decay in the same step, every spike reads as e^(-dt/tau) instead of a clean 1.0. Decay first, bump second.
3. **Crossing the two branches.** post-fires reads *pre's* trace and *strengthens* (+); pre-fires reads *post's* trace and *weakens* (−). Swapping which trace or which sign inverts the whole rule.
4. **Hardcoding A_plus / A_minus** instead of storing the constructor arguments — same silent-disobedience bug as v_rest in Chapter 1.
5. **Window narrower than dt** in modular timing → the pulse can fall between steps and never fire.

---

## 9. Chapter summary

- A synapse can't store spike times, so each neuron carries a single **eligibility trace**: jumps to 1 on its own spike, decays exponentially toward 0 otherwise. It is a recency meter, standing in physically for residual calcium.
- Decay is multiplication by `e^(-dt/tau)` each step — derived from "keep a constant fraction per step." After one tau, ~37% remains.
- The trace's value at the instant another neuron fires equals `e^(-Δt/tau)` — so the trace *computes the spike-timing dependence automatically*, no timestamps stored.
- Two independent clocks: **tau** (electrical, membrane) and **tau_trace** (chemical, learning-window width).
- **STDP rule:** post fires → read pre.trace → strengthen; pre fires → read post.trace → weaken. Sign tracks firing order.
- Verified: causal timing (pre→post) raised the weight 5.00→5.08; reversed timing (post→pre) lowered it 5.00→4.92. Antisymmetric STDP curve, live.
- **Limitation → Chapter 3:** STDP learns correlations but has no notion of good/bad outcomes. A rover on pure STDP would reinforce whatever fires together, including crash-producing pathways. The fix is the **three-factor (dopamine) rule** — gating plasticity by a reward signal.

---
---

# EXAM — Chapter 2

*Closed book. Attempt everything before the answer key at the bottom. ~45–60 min. `e^(-1) ≈ 0.368`, `e^(-0.15) ≈ 0.861`, `e^(-2.35) ≈ 0.095` are given where useful.*

## Part A — Conceptual

**A1.** What single question does an eligibility trace answer, and what are the only two things that ever happen to it?

**A2.** Explain, without equations, why a synapse uses a decaying trace instead of just storing the times its two neurons fired.

**A3.** Name one real physical quantity in a biological neuron that the trace stands in for, and say why that quantity's behavior fits an exponential decay.

**A4.** State the two-branch STDP rule: which spike reads which trace, and the sign of each update.

**A5.** In causal terms, why does "pre fires shortly before post" strengthen a synapse while "post fires shortly before pre" weakens it?

## Part B — The decay term

**B1.** Derive, in words, why the per-step decay multiplier has `dt` on top and `tau` on the bottom in `dt/tau`.

**B2.** With `tau_trace = 10` and `dt = 0.5`, compute the per-step multiplier `e^(-dt/tau)`. (Use e^(-0.05) ≈ 0.951.)

**B3.** A trace is at 1.0 after a spike, `tau_trace = 20`. Give its approximate value at 20 ms, 40 ms, and 60 ms later.

**B4.** Explain in one or two sentences why the trace value at the moment a *second* neuron fires equals `e^(-Δt/tau)`, and why that is the key to the whole scheme.

## Part C — Code

**C1.** Write the two additions to `LIFNeuron.step` that implement the trace (the decay line and the bump line), and state which must come first and why.

**C2.** Write the `Synapse.update_weight` method from memory, including the correct traces and signs.

**C3.** There is a bug below. Identify it, state the runtime symptom, and give the fix.
```python
def step(self, I, t, dt=0.5):
    dv = (dt/self.tau) * (-(self.v - self.v_rest) + self.R*I)
    self.v += dv
    if self.v >= self.v_th:
        self.spike_times.append(t)
        self.v = self.v_reset
        self.trace = 1.0
        return True
    self.trace *= np.exp(-dt / self.tau_trace)
    return False
```

## Part D — Prediction & synthesis

**D1.** A synapse starts at weight 5.0, `A_plus = A_minus = 0.01`. A protocol makes pre fire exactly 3 ms before post, once, and the neurons never fire again. Approximately what is the new weight? (Use e^(-0.15) ≈ 0.861.) Which branch fired and which did not?

**D2.** In the 10-cycle causal experiment the weight rose by ~0.077, not the naive ~0.086. Explain the full mechanism for the difference, referencing both branches and the traces they read.

**D3.** You reverse the timing so post fires 3 ms before pre, same 10 cycles. Predict the direction and approximate magnitude of the weight change, and explain why it mirrors the causal case.

**D4.** (Looking ahead.) Explain in two or three sentences why STDP alone is not enough to make a rover learn to avoid obstacles — what does it lack, and what will Chapter 3 add to fix it?

---
---

# ANSWER KEY

*No peeking until you've attempted everything.*

**Checkpoint 2.A.** It answers "how recently did this neuron fire?" If it fires while the trace is at 0.6, the trace jumps straight to 1.0 (a spike always resets it to 1 regardless of current value).

**Checkpoint 3.A.** ~0.37 at 20 ms (one tau), ~0.135 at 40 ms (two tau).

**Checkpoint 4.A.** e^(-3/20) = e^(-0.15) ≈ 0.86 — high, so a strong weight change.

**Checkpoint 5.A.** Smaller tau_trace. A faster-decaying trace is only still high for spikes very close in time, so only tight coincidences produce a meaningful weight change — a narrow learning window.

**Checkpoint 6.A.** Zero change. When post fires, the strengthen branch reads pre.trace; if pre has never fired, pre.trace is still 0.0, so `A_plus * 0 = 0`. Learning requires both neurons to have spike history.

**Checkpoint 7.A.** Because the weaken branch reads post.trace at the moment *pre* fires, and by then post fired ~47 ms earlier, so its trace has decayed to ~0.095 — whereas the strengthen branch reads pre.trace only 3 ms after pre fired, ~0.86. Same A, very different trace values.

---

**A1.** "How recently did this neuron fire?" The only two things that happen: it jumps to 1.0 when its own neuron fires, and it decays exponentially toward 0 every other step.

**A2.** Storing timestamps would require a growing history buffer and non-local computation the synapse doesn't have access to. A single decaying number captures the one thing that matters — recency — in constant space, and mirrors the actual chemical residue a real synapse can "feel."

**A3.** Calcium concentration (or NMDA-receptor activation, or residual dendritic depolarization). Calcium floods in on a spike and is pumped out gradually over tens of milliseconds, which is naturally an exponential-decay process — fitting e^(-t/tau).

**A4.** When **post** fires, read **pre's** trace and **strengthen** (`weight += A_plus * pre.trace`). When **pre** fires, read **post's** trace and **weaken** (`weight -= A_minus * post.trace`).

**A5.** Pre-before-post is consistent with pre having caused post (Δt > 0), so the connection is rewarded/strengthened. Post-before-pre means post already fired before pre arrived, so pre could not have caused it (Δt < 0); the connection is irrelevant or misleading and is weakened.

---

**B1.** More elapsed time per step should cause more decay, so dt (elapsed time) increases the shrink and belongs on top. A larger tau means slower decay / longer memory, so it must *reduce* the shrink — placing it on the bottom makes dt/tau smaller as tau grows, i.e. less decay per step.

**B2.** e^(-0.5/10) = e^(-0.05) ≈ **0.951** (keeps ~95.1% per step).

**B3.** ~0.368 at 20 ms (one tau), ~0.135 at 40 ms (two tau), ~0.050 at 60 ms (three tau; e^(-3) ≈ 0.0498).

**B4.** If a neuron fires at time t0 (trace → 1) and another fires at t0 + Δt, the first neuron's trace has decayed for exactly Δt, so it equals 1·e^(-Δt/tau) = e^(-Δt/tau). Reading that trace at the second spike therefore hands you the exponential timing-dependence directly, with no timestamp stored — that is the entire mechanism.

---

**C1.**
```python
self.trace *= np.exp(-dt / self.tau_trace)   # decay — every step, first
# ... voltage update and spike check ...
    self.trace = 1.0                          # bump — only inside the spike block
```
Decay must come first so that on a spike step the trace ends at exactly 1.0 and only begins fading the next step; bumping first then decaying would leave a fresh spike reading e^(-dt/tau) instead of 1.0.

**C2.**
```python
def update_weight(self, pre_spiked, post_spiked):
    if post_spiked:
        self.weight += self.A_plus * self.pre.trace
    if pre_spiked:
        self.weight -= self.A_minus * self.post.trace
```

**C3.** The decay line is placed *after* the `if` block, and the spike branch `return True`s before ever reaching it — so on any timestep the neuron spikes, the trace is not decayed, and on non-spike steps it is decayed only after the voltage update. The real problem: on a spike step the code sets trace = 1.0 and returns, which is fine that step, but the decay never runs on spike steps at all, and more importantly the decay was supposed to run on *every* step *before* the bump. **Symptom:** the trace decays on silent steps but the ordering is inconsistent, and because the bump isn't preceded by a decay the semantics drift from the intended "decay-first" model. **Fix:** move `self.trace *= np.exp(-dt/self.tau_trace)` to the very top of `step`, before the voltage update, so it runs unconditionally every step; keep `self.trace = 1.0` inside the spike block.

---

**D1.** Only the **strengthen** branch fires (post fires 3 ms after pre; pre fired first, so when post fires it reads pre.trace ≈ e^(-0.15) ≈ 0.861). The weaken branch would require pre to fire while post's trace is nonzero, which never happens here. New weight ≈ 5.0 + 0.01 × 0.861 = **5.00861**. (The pre-fires event happens before post has ever fired, so its weaken term reads post.trace = 0.)

**D2.** Both neurons spike every cycle, so both branches run. Each cycle: post fires 3 ms after pre → reads pre.trace ≈ e^(-3/20) ≈ 0.86 → +0.0086; then the next pre fires ~47 ms after the previous post → reads post.trace ≈ e^(-47/20) ≈ 0.095 → −0.00095. Net ≈ +0.0077 per cycle, ×10 ≈ +0.077, giving 5.0775 rather than the strengthen-only estimate of 0.086.

**D3.** The weight will **decrease** by approximately the same magnitude (~0.077), landing near 4.92. Reversing the timing swaps which branch sees the large trace and which sees the small one: now post fires first, so when pre fires 3 ms later it reads post.trace ≈ 0.86 → large weaken; the small strengthen term comes from post reading the long-decayed pre.trace. The STDP curve is antisymmetric about Δt = 0, so equal-and-opposite timing gives an equal-and-opposite weight change.

**D4.** STDP is unsupervised: it strengthens whatever fires in causal sequence, with no notion of whether the outcome was good or bad. A rover running pure STDP would reinforce the pathways that fire together even when they drive it into a wall. Chapter 3 adds a third factor — a dopamine/reward signal — that gates plasticity, so weight changes are only consolidated when the outcome is favorable (obstacle avoided) and pruned when it is not (collision). That converts correlation-learning into goal-directed learning.

---

*End of Chapter 2. Next: the three-factor rule — adding dopamine so the network learns not just what fires together, but what's worth doing.*
