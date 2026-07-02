import numpy as np 

class LIFNeuron:

    def __init__(self, tau=20.0, v_rest = -65.0, v_th=-50.0, v_reset=-70.0, R=10.0, tau_trace=20.0):

        self.tau = tau 
        self.v_rest = v_rest
        self.v_th = v_th 
        self.v_reset = v_reset
        self.R = R
        self.v = v_rest 

        self.spike_times = [] 

        self.tau_trace = tau_trace

        self.trace = 0.0  

    def step(self, I, t, dt=0.5):

        # decay trace 

        self.trace *= np.exp(-dt / self.tau_trace)

        dv = (dt/self.tau) * (-(self.v - self.v_rest) + self.R*I)

        self.v += dv 

        if self.v >= self.v_th:

            self.spike_times.append(t)
            self.v = self.v_reset

            self.trace = 1.0 # snap back to 1 on spike 

            return True 
        else:
            return False 
        
class Synapse: 

    def __init__(self, pre:LIFNeuron, post:LIFNeuron, weight, A_plus=0.01, A_minus=0.01):

        self.pre = pre 
        self.post = post 
        self.weight = weight 

        self.A_plus = A_plus
        self.A_minus = A_minus

    def get_current(self, pre_spiked:bool):

        if pre_spiked:

            return self.weight
        else:
            return 0.0
        
    def update_weight(self, pre_spiked:bool, post_spiked:bool):

        if post_spiked:

            self.weight += self.A_plus * self.pre.trace

        if pre_spiked:

            self.weight -= self.A_minus * self.post.trace 


'''

# Scratch
pre = LIFNeuron()
post = LIFNeuron()
dt = 0.5
syn = Synapse(pre, post, weight=60.0)

for t in np.arange(0,200, dt):

    I_pre = 0.0 if t < 20 else 2.00

    pre_spiked = pre.step(I_pre, t, dt)

    I_syn = syn.get_current(pre_spiked)

    if pre_spiked:

        print(f"t = {t}, post_v = {post.v}")

        post.step(I_syn, t, dt)
'''

''' 
# Scratch

n = LIFNeuron()
dt = 0.5

for t in np.arange(0, 60, dt):

    I = 30.0 if t < 2.0 else 0.0 

    n.step(I, t, dt)

    if t % 5 < dt:

        print(f"t={t:1f}    trace={n.trace:.3f}")

print(n.spike_times)
'''

pre = LIFNeuron()
post = LIFNeuron()
syn = Synapse(pre, post, weight=5.0)
dt = 0.5

print(f"start weight: {syn.weight:.4f}")

for t in np.arange(0,500, dt):

    I_pre = 40.0 if (t%50) < 1.0 else 0.0 

    # I_pre = 40.0 if (t%50) >= 3.0 and (t%50) < 4.0 else 0.0 

    I_post = 40.0 if (t%50) >= 3.0 and (t%50) < 4.0 else 0.0 

    # I_post = 40.0 if (t%50) < 1.0 else 0.0 

    pre_spiked = pre.step(I_pre, t, dt)

    post_spiked = post.step(I_post, t, dt)

    syn.update_weight(pre_spiked, post_spiked)

print(f"end weight: {syn.weight:.4f}")
