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
attn_matrices = ["c_q", "c_k", "c_v"]
import itertools
for r in range(len(attn_matrices) + 1):
    for subset in itertools.combinations(attn_matrices, r):
        a=list(subset)
        # if len(a)>0:
        #     arr=torch.load(f"{a}_StiefelAdam_LOSS.pt")
        #     plt.plot(arr,label=f"{a}_stiefeladam")
plt.plot(np.load('/Users/christopherlinder/Desktop/stiefel-nanochat/fullrun_muon.npy'), color='green', label='Muon')
plt.plot(np.load('/Users/christopherlinder/Desktop/stiefel-nanochat/fullrun_stiefeladam.npy'), color='red', label='StiefelAdam')
plt.xlabel('iter')
plt.ylabel('Loss')
plt.title(rf'Muon vs StiefelAdam(WQ, WK)-Transformer')
plt.legend()
plt.show()

a=[]
b=[]
# for i in [1,2,3]:
#     arr=np.load(f"/Users/christopherlinder/Desktop/stiefel-nanochat/12_6961_S{i}_loss_arr.npy")
#     a.append(arr)
#     arr=np.load(f"/Users/christopherlinder/Desktop/stiefel-nanochat/12_6961_CS{i}_loss_arr.npy")
#     b.append(arr)
#     #plt.plot(arr,label=f"{lr}_MuonAdamW",color='green')
# meana=np.mean(a,axis=0)
# stda=np.std(a,axis=0)
# meanb=np.mean(b,axis=0)
# stdb=np.std(b,axis=0)

# plt.plot(meanb,label='Cmean-time=3554.81',color='red')
# plt.fill_between(np.arange(b[0].shape[0]),meanb-stdb,meanb+stdb,color='red')
# arr=np.load('/Users/christopherlinder/Desktop/stiefel-nanochat/12_6961_MUON_loss_arr.npy')
# plt.plot(arr,label='Muon-time=3516.5',color='green')

# plt.plot(meana,label='Smean_time=3542.38',color='blue')
# plt.fill_between(np.arange(a[0].shape[0]),meana-stda,meana+stda,color='blue')

# a=torch.load('STIEFEL-0_LOSS_100.pt')
# c=torch.load('STIEFELAdam-0_LOSS_100.pt')
# b=torch.load('data/losses/LOSS_100_default.pt')
# print(a[:4], b[:4])

#plt.plot(torch.load("MuonAdamW_LOSS.pt"),color='green',label="muon")

# #plt.plot(b,color='blue',label="default")
# plt.plot(c,color='green',label="stiefeladam")


# plt.xlabel("iter_num")
# plt.ylabel('Cross-Entropy_Loss')
# plt.legend()
# plt.show()

import matplotlib.pyplot as plt

a=torch.load('muon_transformer_h.5_attn_c_q_weight.pt')
b=torch.load('stiefeladam_transformer_h.5_attn_c_q_weight.pt')
q=torch.load('stiefeladam_Q_transformer_h.5_attn_c_q_weight.pt')
k=torch.load('stiefeladam_Q_transformer_h.5_attn_c_k_weight.pt')

print(a.shape, q.shape)

ah1=a[:,:1536//12]
qh1=q[:,:1536//12]
Q, R=torch.linalg.qr(ah1)
print(ah1.shape, Q.shape, R.shape)
print(torch.linalg.norm(qh1.T@qh1, ord='fro'))
print(torch.linalg.norm(qh1-Q, ord='fro'))

Q, R=torch.linalg.qr(a)
print(torch.linalg.norm(q-Q, ord='fro'))
print(q)
print(Q)
print(q-Q)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
imG = axes[0].imshow(q, cmap='viridis')
imP = axes[1].imshow(Q, cmap='viridis')
axes[0].set_title('G (step 0)')
axes[1].set_title('P (step 0)')

fig.colorbar(imG, ax=axes[0], fraction=0.046, pad=0.04)
fig.colorbar(imP, ax=axes[1], fraction=0.046, pad=0.04)

# Display the figure
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
# print(torch.linalg.norm(g.T@g,ord='fro'))
# print(torch.sum(g@g.T), torch.trace(g@g.T))
# print(g@g.T, g.T@g)