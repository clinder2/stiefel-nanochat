import torch
import numpy as np
import matplotlib.pylab as plt

# a=[torch.load('LOSS_100_default.pt')]
# b=[torch.load('LOSS_100_within.pt')]
# for i in range(1,5):
#     a.append(torch.load(f'DEFAULT-{i}_LOSS_100.pt'))
#     b.append(torch.load(f'ORTHO-WITHIN-{i}_LOSS_100.pt'))

# meana=np.mean(a,axis=0)
# meanb=np.mean(b,axis=0)
# stda=np.std(a,axis=0)
# stdb=np.std(b,axis=0)

# plt.plot(meana,color='blue',label="default")
# plt.fill_between(np.arange(a[0].shape[0]),meana-stda,meana+stda,color='red')
# plt.plot(meanb,color='green',label='orthog_within_heads(q,k_separate)')
# plt.fill_between(np.arange(a[0].shape[0]),meanb-stdb,meanb+stdb,color='orange')
a=torch.load('STIEFEL-0_LOSS_100.pt')
c=torch.load('STIEFELAdam-0_LOSS_100.pt')
b=torch.load('data/losses/LOSS_100_default.pt')
print(a[:4], b[:4])
plt.plot(a,color='red',label="stiefel")
#plt.plot(b,color='blue',label="default")
plt.plot(c,color='green',label="stiefeladam")
plt.xlabel("iter_num")
plt.ylabel('Cross-Entropy_Loss')
plt.legend()
plt.show()

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
ns_steps=5
g=torch.randn((16,16))

X = g.bfloat16()
X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
if g.size(-2) > g.size(-1): # Tall matrix
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
else: # Wide matrix (original math)
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
g = X

g=g.to(torch.float32)
print(torch.linalg.norm(g.T@g,ord='fro'))
print(torch.sum(g@g.T), torch.trace(g@g.T))
print(g@g.T, g.T@g)