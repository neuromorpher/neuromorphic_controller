# Spiking Neurons from Scratch — Chapter 1

### The Leaky Integrate-and-Fire Neuron, the Synapse, and the Seed of Plasticity

*A study guide built from your own work. Every number in here is one you produced and verified in the session — nothing is hand-waved.*

---

## How to use this document

Read it once straight through for the narrative. Then read it a second time with a pen, and **redo every worked example yourself before looking at the worked-out arithmetic.** The single most important habit from our sessions was *predict → run → check*. This guide is built to make you keep doing that. The exam at the end has no value if you read the answers first — attempt everything cold, then grade yourself.

There are **Checkpoint** boxes scattered through the text. Stop at each one and answer it out loud or on paper before continuing. If you can't, that section hasn't landed yet — reread it.

---

## Learning objectives

By the end of this chapter you should be able to, *from memory*:

1. Write the leaky integrate-and-fire (LIF) differential equation and explain what every term does physically.
2. Discretize that equation with the Euler method and explain why the discrete form looks the way it does.
3. Implement a `LIFNeuron` class and a `Synapse` class in plain Python, with no libraries and no autocomplete.
4. Predict whether a neuron under constant input will spike — *before* running it — and predict its firing rate.
5. Explain why a single weak synapse cannot drive a postsynaptic neuron, in terms of the leak time constant.
6. Derive the skipped-beat (subharmonic) firing pattern analytically, not just observe it.
7. State why the spike-time difference Δt is the only quantity a biological synapse could possibly learn from, and why that points toward STDP.

---

## 1. The big picture (why we're doing any of this)

Standard artificial neurons are "point neurons": they take a weighted sum of inputs, push it through an activation function, and emit a single number. Time doesn't really exist for them — everything happens in one instantaneous forward pass, and learning happens through *global* error correction (backpropagation), where an error signal computed at the output is propagated backward through the entire network.

A biological neuron is a different kind of object. It is a small dynamical system that evolves in continuous time. It accumulates input, leaks charge, and occasionally emits a discrete event — a **spike**. It does not have access to a global error signal. Whatever it learns, it must learn from information physically available at its own connections.

The long-term project — a closed-loop neuromorphic controller for a physical rover — is built entirely out of these spiking units. This chapter is the first brick: getting *one* neuron, then *two connected* neurons, to behave correctly and understanding exactly why they behave the way they do. Everything later (STDP, dopamine-modulated plasticity, population coding) is built on top of the dynamics in this chapter. If these foundations are shaky, nothing above them will make sense.

---

## 2. The leaky integrate-and-fire neuron

### 2.1 The intuition: a leaky bucket

Picture a bucket with a small hole in the bottom.

- **Input current** is water pouring in from above.
- **The leak** is water draining out of the hole. Crucially, the drain is *faster the fuller the bucket is* — it's proportional to how far the water sits above its natural resting level.
- If you pour in fast enough, the level rises despite the leak.
- When the level crosses a **threshold**, the neuron **spikes**, and the bucket is instantly emptied to a **reset level**, below where it started. Then it begins filling again.

That's the entire model. Everything below is just that picture written precisely.

### 2.2 The equation

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)$$

Term by term:

| Symbol | Meaning | Our value |
|--------|---------|-----------|
| $V$ | membrane potential right now (the water level) | starts at $V_{rest}$ |
| $V_{rest}$ | the level the neuron settles to with zero input | $-65$ mV |
| $\tau$ (tau) | time constant — bigger $\tau$ means it leaks/forgets *more slowly* | $20$ ms |
| $R$ | membrane resistance — scales how much voltage a given input current produces | $10$ |
| $I(t)$ | input current at this instant | varies |
| $V_{th}$ | threshold; cross it and the neuron spikes | $-50$ mV |
| $V_{reset}$ | the level $V$ snaps to right after a spike | $-70$ mV |

The two terms on the right-hand side are doing two different jobs:

- $-(V - V_{rest})$ is **the leak**. When $V$ is above rest, this term is negative and pulls $V$ back down. When $V$ is below rest, it's positive and pulls $V$ back up. It always drags $V$ *toward* rest, and harder the further away $V$ is.
- $R \cdot I(t)$ is **the drive**. This is the input pushing $V$ around.

> **⚠️ The single most dangerous subtlety in this whole chapter.**
> The right-hand side is **two separate terms added together**:
> $$-(V - V_{rest}) \;+\; R \cdot I$$
> It is **NOT** $-\big((V - V_{rest}) + R \cdot I\big)$. The negative sign belongs *only* to the leak term. The drive term $R \cdot I$ is added on, unnegated. That unnegated drive term is the entire reason input current makes voltage *rise* instead of fall. If you internalize "the whole bracket gets negated," you will eventually produce a neuron that does the exact opposite of what it should — and the bug will look like correct math.

### 2.3 The spike-and-reset rule

The differential equation alone would just let $V$ rise and asymptote. The spiking behavior is an extra rule layered on top:

> Whenever $V \geq V_{th}$: record a spike at the current time, then immediately set $V = V_{reset}$.

This is what makes it *integrate-and-fire* rather than just *integrate*.

### 2.4 From continuous to discrete: Euler's method

A computer can't handle continuous time. We chop time into small steps of size $dt$ and update $V$ step by step. The Euler method replaces $\frac{dV}{dt}$ with $\frac{\Delta V}{dt}$ and solves for the change:

$$dV = \frac{dt}{\tau}\Big(-(V - V_{rest}) + R \cdot I\Big)$$

Then on each timestep:

```
dV = (dt / tau) * ( -(V - V_rest) + R*I )
V  = V + dV
if V >= V_th:
    record spike
    V = V_reset
```

That is the complete numerical neuron. Smaller $dt$ = more accurate but slower; we used $dt = 0.5$ ms.

> **Checkpoint 2.A.** Without looking up, write the LIF equation, then the Euler update. Then say in one sentence what would physically happen to the neuron's behavior if you doubled $\tau$.

---

## 3. The code: `LIFNeuron`

```python
class LIFNeuron:
    def __init__(self, tau=20.0, v_rest=-65.0, v_th=-50.0, v_reset=-70.0, R=10.0):
        self.tau = tau
        self.v_rest = v_rest
        self.v_th = v_th
        self.v_reset = v_reset
        self.R = R
        self.v = self.v_rest          # current voltage starts at rest
        self.spike_times = []         # log of when this neuron fired

    def step(self, I, t, dt=0.5):
        dV = (dt / self.tau) * ( -(self.v - self.v_rest) + self.R * I )
        self.v += dV
        if self.v >= self.v_th:
            self.spike_times.append(t)
            self.v = self.v_reset
            return True
        else:
            return False
```

Two design points worth holding onto:

- **Every parameter is stored from the argument** (`self.v_rest = v_rest`), never hardcoded (`self.v_rest = -65.0`). Hardcoding silently ignores whatever the caller passes in, and Python gives you *no error* — the neuron just quietly disobeys you. This is one of the worst bug classes there is.
- **`step` returns a boolean** saying whether the neuron fired this timestep. That return value is what lets us wire neurons together: one neuron's "did I spike?" becomes another neuron's input.

---

## 4. Driving one neuron, and predicting before running

### 4.1 The steady-state ceiling

Set $\frac{dV}{dt} = 0$ in the LIF equation (the voltage the neuron would drift to and stay at, if it never spiked):

$$0 = -(V_{ss} - V_{rest}) + R \cdot I \quad\Longrightarrow\quad \boxed{V_{ss} = V_{rest} + R \cdot I}$$

This is the most useful prediction tool you have. **Before running anything**, compute $V_{ss}$ and compare it to $V_{th}$:

- If $V_{ss} > V_{th}$: the neuron *will* spike repeatedly. (It's trying to settle at a level above threshold, so it keeps crossing it.)
- If $V_{ss} < V_{th}$: the neuron will rise, asymptote below threshold, and **never spike**, no matter how long you wait.

**Worked example (our run).** $V_{rest} = -65$, $R = 10$, $I = 2$:
$$V_{ss} = -65 + (10)(2) = -45 \text{ mV}$$
$-45 > -50$, so it spikes. ✔ And indeed our run produced spikes.

### 4.2 The driver loop

```python
import numpy as np

n = LIFNeuron()
dt = 0.5
voltages = []

for t in np.arange(0, 200, dt):
    I = 0.0 if t < 20 else 2.0     # quiet, then current switches on
    n.step(I, t, dt)
    voltages.append(n.v)

print(n.spike_times)        # [47.0, 79.0, 111.0, 143.0, 175.0]
print(len(n.spike_times))   # 5
```

### 4.3 Constant input → constant firing rate

Look at the spike times: `47, 79, 111, 143, 175`. After the first spike, the gaps are **exactly 32 ms every time** ($79-47$, $111-79$, …).

This is not a coincidence and it's worth understanding deeply. After every spike, the neuron resets to the *same* $V_{reset} = -70$, then faces the *same* constant input current, so it always takes the *same* amount of time to climb back to threshold. **Constant input produces a single, clean, well-defined firing rate.** The first interval (from $t=20$ when current switched on, to the first spike at $t=47$) is different only because the neuron started at rest, not at $V_{reset}$.

Hold onto this: one neuron's firing rate being a single number is the atomic unit that **population coding** will later combine across hundreds of neurons.

> **Checkpoint 4.A.** A neuron has $V_{rest}=-65$, $V_{th}=-50$, $R=10$. You give it constant $I = 1.0$. Does it spike? Show the one-line calculation. *(Answer at the very bottom — but compute it first.)*

---

## 5. The synapse

### 5.1 What a synapse is

A synapse is the connection from one neuron (the **presynaptic**, or "pre") to another (the **postsynaptic**, "post"). When pre fires, post receives an extra jolt of input current, scaled by a single number: the **weight**.

That weight is the most important number in the entire project. *Every* learning rule we will ever build — STDP, the dopamine three-factor rule, all of it — works by modifying this one number. That's why it lives in its own object even though, right now, it just sits there as a constant.

### 5.2 The code: `Synapse`

```python
class Synapse:
    def __init__(self, pre: LIFNeuron, post: LIFNeuron, weight):
        self.pre = pre
        self.post = post
        self.weight = weight

    def get_current(self, pre_spiked: bool):
        if pre_spiked:
            return self.weight
        else:
            return 0.0
```

The synapse delivers its weight as current *only on the single timestep where pre fired*, and zero otherwise. This "brief pulse, then nothing" shape is critical to everything in Section 6.

### 5.3 Wiring two neurons together

```python
pre  = LIFNeuron()
post = LIFNeuron()
syn  = Synapse(pre, post, weight=60.0)

for t in np.arange(0, 200, dt):
    I_pre = 0.0 if t < 20 else 2.0
    pre_spiked = pre.step(I_pre, t, dt)     # drive the pre neuron
    I_syn = syn.get_current(pre_spiked)     # pre's spike -> current
    post.step(I_syn, t, dt)                 # that current drives post

print(pre.spike_times)
print(post.spike_times)
```

Note the data flow: external current drives **pre**; pre's spike becomes **post's** input through the synapse. Post receives *nothing* except these brief synaptic pulses.

---

## 6. Two-neuron dynamics — the heart of the chapter

### 6.1 Why a weak synapse fails

First we tried `weight = 15`. A single pulse arriving while post sits at rest produces:
$$dV = \frac{dt}{\tau} \cdot R \cdot w = \frac{0.5}{20}\cdot 10 \cdot 15 = 3.75 \text{ mV}$$

So one pulse lifts post from $-65$ to about $-61.25$. There are 15 mV to cover to reach threshold. It's tempting to think "a few more pulses will accumulate and get there." **They won't, and the reason is the whole point.**

Between pre-spikes, post is **not frozen**. The leak term $-(V - V_{rest})$ is active on *every* timestep, even when $I_{syn} = 0$. So after each pulse lifts post by 3.75 mV, the leak immediately starts dragging it back toward $-65$. With pre firing every ~32 ms and $\tau = 20$ ms, the gap between pulses is *longer than the time constant* — so most of each boost decays away **before the next pulse arrives.** The pulses don't stack. Each one is nearly forgotten before the next shows up.

We confirmed this directly: post's voltage jumped ~3.7 mV at each pre-spike ($t = 79, 111, 143, 175$) and decayed back to about $-64$ in every gap. The peak after pulse 5 looked essentially identical to the peak after pulse 2. With this weight, post would *never* fire — not with 5 pulses, not with 500.

**The lesson:** whether inputs accumulate depends on the relationship between the inter-pulse interval and $\tau$. Pulses faster than $\tau$ build on each other; pulses slower than $\tau$ each die in isolation.

### 6.2 Finding the weight that works

What weight makes a *single* pulse cross the entire 15 mV gap in one step? Set the pulse-from-rest equal to the gap (leak ≈ 0 because we start at rest):
$$\frac{dt}{\tau}\cdot R \cdot X = 15 \quad\Longrightarrow\quad \frac{0.5}{20}\cdot 10 \cdot X = 15 \quad\Longrightarrow\quad 0.25 X = 15 \quad\Longrightarrow\quad \boxed{X = 60}$$

With `weight = 60`, we reran and got:

```
pre.spike_times  = [47, 79, 111, 143, 175]   # 5 spikes
post.spike_times = [47, 111, 175]            # only 3 — it skipped 79 and 143!
```

### 6.3 The skipped-beat (subharmonic) pattern

Pre fires 5 identical pulses; post fires on only 3 of them, in a regular fire–skip–fire–skip rhythm. Why would two of five *identical* pulses fail? We printed post's voltage at the instant each pulse arrived:

```
t = 47:   post_v = -65.00    -> fires
t = 79:   post_v = -66.01    -> fails
t = 111:  post_v = -62.16    -> fires
t = 143:  post_v = -66.01    -> fails
t = 175:  post_v = -62.16    -> fires
```

The pulse only succeeds when post is sitting at the **−62 "peak"**, and fails when it's at the **−66 "valley."** Here is the full mechanism, and you can derive every number analytically.

**The recovery between events follows exponential decay toward rest:**
$$V(t) = V_{rest} + (V_0 - V_{rest})\,e^{-t/\tau}$$

**After a real spike**, post resets deep to $V_{reset} = -70$. Over the 32 ms gap (which is $32/20 = 1.6$ time constants), it recovers to:
$$V = -65 + (-70 - (-65))\,e^{-1.6} = -65 + (-5)(0.202) = -66.01 \text{ mV}$$
That's our "valley." It's ~1 mV *below* rest — it hasn't even finished recovering to $-65$. A pulse from here reaches:
$$-66.01 + \tfrac{0.5}{20}\big(-(-66.01+65) + 10\cdot 60\big) = -66.01 + 0.025(1.01 + 600) = -50.99 \text{ mV}$$
$-50.99 < -50$. **Just barely fails**, by about 1 mV.

A **failed pulse never triggers a reset**, so post is left sitting high, near $-51$. Over the *next* 32 ms it decays from there (not from $-70$):
$$V = -65 + (-50.99 - (-65))\,e^{-1.6} = -65 + (14.01)(0.202) = -62.17 \text{ mV}$$
That's our "peak." A pulse from $-62.16$ comfortably clears threshold → **fires** → resets to $-70$ → undershoots → fails → recovers shallow → fires → … forever.

This is a **deterministic, self-sustaining rhythm**: deep reset → undershoot → skip → shallow recovery → fire → deep reset → … Nothing about it changes cycle to cycle, so it locks in. Real neurons show exactly this kind of rate-dividing behavior; it's not a simulation artifact.

> **Checkpoint 6.A.** In your own words: why does the *failed* pulse make the *next* pulse more likely to succeed? (Hint: what does "no reset" do to post's starting voltage for the following gap?)

### 6.4 What you actually have now

Two connected neurons with **real relative spike timing** between them: pre on a known schedule, post firing on a subset of those instants. The quantity
$$\Delta t = t_{post} - t_{pre}$$
— the gap between a pre-spike and a post-spike — is the exact number that all of plasticity is built on. That's Section 7.

---

## 7. Why Δt? (the seed of STDP)

This is the conceptual hinge to the next chapter, so make sure it lands.

**A synapse can only learn from what's physically present at its own location.** Sitting between exactly two neurons, the only two events that physically happen *at that spot* are:

1. The moment **pre** fired (neurotransmitter is released there).
2. The moment **post** fired (biologically, this sends a signal *backward* down the dendrite, sweeping past that same synapse).

Those are the only two timestamps in the world that a synapse has access to. There is **no global error signal, no loss function, no instruction from elsewhere** in the network. Just two local times. So whatever the synapse learns, it must be a function of those two times — and the only meaningful thing you can build from two timestamps is their difference, $\Delta t = t_{post} - t_{pre}$.

The logic placed on $\Delta t$ is **causal inference**:

- **$\Delta t$ small and positive** (pre fired *just before* post): consistent with pre having *helped cause* post to fire. → **strengthen** the synapse. ("What I just did seems to have mattered.")
- **$\Delta t$ negative** (pre fired *after* post already spiked): pre cannot have caused something that already happened. → **weaken** the synapse.

The **sign** of $\Delta t$ does the causal direction (did I come before or after?). The **magnitude** does the confidence weighting (closer in time → more likely causal → bigger update).

This is exactly what's meant by *plasticity via causality* rather than *global error correction*. Backpropagation requires a network-wide error signal delivered to every weight — a privilege no biological synapse has. STDP (spike-timing-dependent plasticity) is what's left when you ask: *what could a synapse actually learn, using only the information physically available to it?* The answer is the sign and size of $\Delta t$.

> **Checkpoint 7.A.** State, without notes, the two timestamps a synapse has local access to, and what the *sign* of their difference is used to infer.

---

## 8. Common pitfalls (every one of these actually bit us)

1. **Hardcoding instead of storing parameters.** `self.v_rest = -65.0` instead of `self.v_rest = v_rest` silently ignores the caller. No error. The object just disobeys.
2. **Sign errors in the LIF terms.** `-(V - V_rest) + R*I` is two terms; the negation is *only* on the leak. Wrapping the whole thing in the negative inverts the drive.
3. **A wrong-signed constant.** We briefly had `v_rest = 65.0` instead of `-65.0`. A neuron resting at +65 is permanently above threshold and spikes every single step. No error message — just nonsense behavior.
4. **A loop/condition that never resets.** A print guarded by `t > some_value` is a *floor*, not a *window* — once true it stays true, and you get runaway output. To print a window you need both a lower *and* upper bound.
5. **Editing without saving.** An `IndexError` (or any "but I just fixed that!") is very often a file you changed but didn't save. Check this first.

---

## 9. Chapter summary

- A LIF neuron is a leaky bucket: it integrates input, leaks toward rest proportionally to its distance from rest, and fires + resets when it crosses threshold.
- The model is $\tau\,dV/dt = -(V - V_{rest}) + R\,I$, with a spike-and-reset rule on top. The leak and drive are **separate** terms.
- Discretize with Euler: $dV = (dt/\tau)(-(V-V_{rest}) + R I)$.
- Predict before you run: steady state is $V_{ss} = V_{rest} + R I$. If $V_{ss} > V_{th}$, it spikes; otherwise it never does.
- Constant input → constant inter-spike interval → a single clean firing rate (the unit of population coding).
- A synapse delivers its **weight** as a brief one-step current pulse when pre fires. The weight is what all future learning modifies.
- Whether pulses accumulate depends on inter-pulse interval vs. $\tau$. Slow pulses (gap > $\tau$) decay before the next arrives and can't add up.
- Deep reset + shallow recovery can produce a deterministic **skipped-beat** firing pattern, fully derivable from exponential decay.
- $\Delta t = t_{post} - t_{pre}$ is the only locally available learning signal; its sign infers causal direction and its magnitude infers confidence. That's the seed of STDP.

---
---

# EXAM — Chapter 1

*Closed book. Attempt everything before checking the answer key at the very bottom. Budget ~60 minutes. Calculators allowed; $e^{-1.6} \approx 0.202$ is given where needed.*

## Part A — Conceptual (short answer)

**A1.** In the LIF equation, what does the term $-(V - V_{rest})$ do when $V$ is *above* rest? When *below*? (One sentence each.)

**A2.** Explain in your own words why $R \cdot I$ must be added *outside* the negation, not inside it. What would go wrong physically if it were inside?

**A3.** Why does constant input current produce a constant inter-spike interval after the first spike?

**A4.** A synapse has access to exactly two timestamps. Name them. What is the difference between them used to infer, and what does the *sign* of that difference tell you?

**A5.** Distinguish, in two or three sentences, STDP's notion of "learning via causality" from backpropagation's "global error correction." Why can't a biological synapse use the backprop approach?

## Part B — Prediction & calculation

**B1.** A LIF neuron has $V_{rest} = -70$, $V_{th} = -54$, $R = 8$. You apply constant $I = 1.5$. (a) Compute the steady-state voltage $V_{ss}$. (b) Will it spike? (c) Justify in one sentence.

**B2.** Using $V_{rest}=-65$, $V_{th}=-50$, $R=10$, $\tau=20$, $dt=0.5$: a synapse delivers a single pulse to a neuron sitting *exactly at rest*. What weight $w$ is required for that one pulse to bring the neuron exactly to threshold? Show the algebra.

**B3.** Same parameters as B2. A neuron is sitting at $V = -68$ mV (below rest) when a pulse of weight $w = 40$ arrives. Compute $dV$ for that timestep. Be careful with the sign of the leak term. Does it cross threshold?

**B4.** A neuron resets to $V_{reset} = -70$ and then receives *no* input for 40 ms. Using $V(t) = V_{rest} + (V_0 - V_{rest})e^{-t/\tau}$ with $V_{rest} = -65$, $\tau = 20$, compute its voltage at the end of the 40 ms. (You may use $e^{-2} \approx 0.135$.)

## Part C — Code

**C1.** Write the `LIFNeuron.__init__` method from memory, with the same five parameters and defaults used in this chapter, plus the two extra instance variables. No peeking.

**C2.** Write the `step` method from memory. It must compute $dV$, update voltage, handle the spike-and-reset, and return the right boolean.

**C3.** There is a bug in the code below. Identify it, explain what *symptom* it produces at runtime (does it crash? misbehave silently?), and give the fixed line.
```python
class LIFNeuron:
    def __init__(self, tau=20.0, v_rest=-65.0, v_th=-50.0, v_reset=-70.0, R=10.0):
        self.tau = tau
        self.v_rest = -65.0
        self.v_th = v_th
        self.v_reset = v_reset
        self.R = R
        self.v = self.v_rest
        self.spike_times = []
```

## Part D — Synthesis (harder)

**D1.** Explain, in terms of $\tau$ and the inter-pulse interval, why a synapse with `weight=15` failed to ever fire the post neuron in our experiment, even across many pulses. Your answer must reference *why the pulses did not accumulate*.

**D2.** In the skipped-beat experiment (`weight = 60`), post fired on pulses 1, 3, 5 and skipped 2, 4. Reconstruct the mechanism: explain why the pulse arriving right after a successful spike *fails*, and why the very next pulse then *succeeds*. Reference what a reset does vs. what a failed pulse does to the neuron's starting voltage for the following gap.

**D3.** Suppose you *shortened* $\tau$ from 20 ms to 5 ms while keeping everything else (including the ~32 ms pre-firing interval) the same. Qualitatively, would a weak synapse's pulses be *more* or *less* able to accumulate? Explain using the relationship between $\tau$ and the inter-pulse gap.

---
---

# ANSWER KEY

*No peeking until you've attempted everything.*

**Checkpoint 2.A.** Doubling $\tau$ makes the neuron leak/forget more slowly — it integrates input over a longer window and its voltage changes more sluggishly per timestep.

**Checkpoint 4.A.** $V_{ss} = -65 + 10(1.0) = -55$. Since $-55 < -50$ ($V_{th}$), it asymptotes below threshold and **never spikes**.

**Checkpoint 6.A.** A failed pulse triggers no reset, so post is left sitting high (≈ −51 instead of −70). For the following gap it decays from that high position and lands at a much higher "starting" voltage (≈ −62) when the next pulse arrives, so that pulse only has to cover a small remaining distance to threshold and succeeds.

**Checkpoint 7.A.** (1) The time pre fired and (2) the time post fired. The *sign* of $t_{post} - t_{pre}$ infers causal direction — whether pre fired before post (could have caused it → strengthen) or after (could not have → weaken).

---

**A1.** Above rest: the term is negative, pulling $V$ back *down* toward rest. Below rest: positive, pushing $V$ back *up* toward rest. It always drags toward rest, harder the further away.

**A2.** Because the drive must be able to push voltage *up*. If $R\cdot I$ were inside the negation, positive input current would *subtract* from voltage, and the neuron would move away from threshold under excitation — the opposite of a neuron. The negation is conceptually attached only to "distance from rest," i.e. the leak.

**A3.** Every spike resets the neuron to the same $V_{reset}$, after which it faces the same constant input, so the climb back to threshold takes the same time every cycle.

**A4.** The time pre fired and the time post fired. Their difference $\Delta t$ is used to infer causality; the sign tells you whether pre preceded post (consistent with pre causing the spike → strengthen) or followed it (cannot be causal → weaken).

**A5.** STDP updates each synapse using only *local* timing information ($\Delta t$ between its own two neurons) to infer whether that connection was causally useful. Backprop computes a *global* error at the network output and propagates it backward to every weight. A biological synapse has no access to a network-wide error signal or to the activity of distant neurons — only to its own two endpoints — so it cannot implement backprop.

---

**B1.** (a) $V_{ss} = -70 + 8(1.5) = -70 + 12 = -58$ mV. (b) No. (c) $-58 < -54$ ($V_{th}$), so it asymptotes below threshold and never reaches it.

**B2.** $\frac{0.5}{20}\cdot 10 \cdot w = 15$. The gap from $-65$ to $-50$ is 15 mV. $0.25\,w = 15 \Rightarrow w = 60$.

**B3.** Leak term: $-(V - V_{rest}) = -(-68 - (-65)) = -(-3) = +3$. Then
$$dV = \tfrac{0.5}{20}\big(3 + 10\cdot 40\big) = 0.025(3 + 400) = 0.025 \cdot 403 = 10.075 \text{ mV.}$$
New voltage: $-68 + 10.075 = -57.925$. Since $-57.925 < -50$, it does **not** cross threshold.

**B4.** $V = -65 + (-70 - (-65))e^{-40/20} = -65 + (-5)e^{-2} = -65 + (-5)(0.135) = -65 - 0.675 = -65.675$ mV. (It has recovered from −70 to just below rest.)

---

**C1 / C2.** Compare against Section 3 verbatim. Key things that must be correct: every parameter stored *from its argument* (no hardcoding), `self.v = self.v_rest`, `self.spike_times = []`, the $dV$ formula with the leak/drive signs right, voltage update, `>=` threshold comparison, append-then-reset order, and returning `True`/`False` appropriately.

**C3.** The bug is `self.v_rest = -65.0` — it hardcodes the value instead of storing the argument (`self.v_rest = v_rest`). **Symptom:** no crash, no error message; it misbehaves *silently*. Any caller who passes a custom `v_rest` is ignored, and the neuron always rests at −65 regardless. Fix: `self.v_rest = v_rest`.

---

**D1.** Pre fires roughly every 32 ms, but $\tau = 20$ ms. Because the inter-pulse gap (32 ms) is *larger* than the time constant, each 3.75 mV boost decays substantially back toward rest *before the next pulse arrives* — the leak removes most of it during the gap. So the pulses do not stack; each one lands on a neuron that has nearly returned to rest, and the peak after many pulses is essentially the same as after the first. With weight 15 that peak is far below threshold, so it never fires.

**D2.** A successful spike resets post deep to $V_{reset} = -70$, 5 mV below rest. Over the next 32 ms (= 1.6 $\tau$) it only recovers to about −66 (still below rest). A pulse from −66 reaches about −50.99, which is *just* below threshold (−50) — it fails. But a failed pulse causes **no reset**, leaving post sitting high near −51. Over the following gap it decays from that high value down to only about −62. A pulse from −62 clears threshold easily and fires — which resets it deep again, restarting the cycle. So deep-reset pulses fail and failed-pulse-elevated ones succeed, producing the regular fire/skip alternation.

**D3.** With $\tau = 5$ ms, the time constant is now *much shorter* than the 32 ms gap, so each boost decays even faster and more completely between pulses — accumulation becomes *less* possible, not more. (Accumulation improves when $\tau$ is *large* relative to the gap, so that a boost is still substantially present when the next pulse arrives. Shrinking $\tau$ makes the neuron forget faster and hurts summation.)

---

*End of Chapter 1. Next chapter: turning $\Delta t$ into an actual weight-update rule — STDP.*
