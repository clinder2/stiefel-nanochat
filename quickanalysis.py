import torch
import matplotlib.pyplot as plt
import pickle

w1=torch.load("/storage/project/r-mtao8-0/clinder9/.cache/nanochat/speedruns/base_checkpoints/FULLRUN_MUON/model_002088.pt", map_location=torch.device('cpu'))

w2=torch.load("/storage/project/r-mtao8-0/clinder9/.cache/nanochat/speedruns/base_checkpoints/FULLRUN_STIEFELADAM_Q/model_002088.pt", map_location=torch.device('cpu'))


#torch.save(w1['transformer.h.5.attn.c_q.weight'], "muon_transformer_h.5_attn_c_q_weight.pt")

torch.save(w2['transformer.h.5.attn.c_q.weight'], "stiefeladam_Q_transformer_h.5_attn_c_q_weight.pt")
torch.save(w2['transformer.h.5.attn.c_k.weight'], "stiefeladam_Q_transformer_h.5_attn_c_k_weight.pt")

# fig, axes = plt.subplots(1, 2, figsize=(8, 8))
# imW1 = axes[0].imshow(w1['transformer.h.5.attn.c_q.weight'], cmap='viridis')
# imW2 = axes[1].imshow(w2['transformer.h.5.attn.c_q.weight'], cmap='viridis')
# axes[0].set_title('muon layer 5 Q weight')
# axes[1].set_title('stiefeladam layer 5 Q weight')

# fig.colorbar(imW1, ax=axes[0], fraction=0.046, pad=0.04)
# fig.colorbar(imW2, ax=axes[1], fraction=0.046, pad=0.04)

# with open('muon_vs_stiefeladam_Qweight.pkl', 'wb') as f:
#     pickle.dump(fig, f)

# print(w1.keys())
# for k in w1.keys():
#     if len(w1[k].shape)<2:
#         print(f"param_name: {k}, w1: {torch.linalg.norm(w1[k], ord=2)}, w2: {torch.linalg.norm(w2[k], ord=2)}, diff: {torch.linalg.norm(w1[k]-w2[k], ord=2)}")
#     else:
#         print(f"param_name: {k}, w1: {torch.linalg.norm(w1[k], ord='fro')}, w2: {torch.linalg.norm(w2[k], ord='fro')}, diff: {torch.linalg.norm(w1[k]-w2[k], ord='fro')}")