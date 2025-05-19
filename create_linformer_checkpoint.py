import torch
import os
import numpy as np

# Paths
orig_ckpt_path = r"checkpoint/ucf-crime_s3r_i3d_best.pth"
new_ckpt_path = r"checkpoint/Linformer_chkpoint.pth"
macro_dict_path = r"dictionary/ucf-crime/ucf-crime_dictionaries.taskaware.omp.100iters.50pct.npy"

# Linformer config
feature_dim = 2048
embed_dim = 512  # matches original query/cache embedding output
k = 16          # Linformer projection length

# Load macro dictionary to get S
macro = np.load(macro_dict_path)
seq_len = macro.shape[0]  # S = number of dictionary atoms

# Load original checkpoint
ckpt = torch.load(orig_ckpt_path, map_location='cpu')
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    state_dict = ckpt

# Remove old enNormal attention weights
to_remove = [
    'en_normal.en_normal_module.query_embedding.weight',
    'en_normal.en_normal_module.query_embedding.bias',
    'en_normal.en_normal_module.cache_embedding.weight',
    'en_normal.en_normal_module.cache_embedding.bias',
    'en_normal.en_normal_module.value_embedding.weight',
    'en_normal.en_normal_module.value_embedding.bias',
]
for key in to_remove:
    if key in state_dict:
        del state_dict[key]

# Add new Linformer weights (randomly initialized)
import collections
new_weights = collections.OrderedDict()

def rand_tensor(shape):
    return torch.randn(*shape)

def rand_bias(shape):
    return torch.zeros(*shape)

# Q, K, V projections: [embed_dim, feature_dim] and [embed_dim]
new_weights['en_normal.en_normal_module.linformer_q_proj.weight'] = rand_tensor((embed_dim, feature_dim))
new_weights['en_normal.en_normal_module.linformer_q_proj.bias'] = rand_bias((embed_dim,))
new_weights['en_normal.en_normal_module.linformer_k_proj.weight'] = rand_tensor((embed_dim, feature_dim))
new_weights['en_normal.en_normal_module.linformer_k_proj.bias'] = rand_bias((embed_dim,))
new_weights['en_normal.en_normal_module.linformer_v_proj.weight'] = rand_tensor((embed_dim, feature_dim))
new_weights['en_normal.en_normal_module.linformer_v_proj.bias'] = rand_bias((embed_dim,))
# Linformer projections for keys and values: [seq_len, k]
new_weights['en_normal.en_normal_module.linformer_E_proj.weight'] = rand_tensor((int(seq_len), int(k)))
new_weights['en_normal.en_normal_module.linformer_F_proj.weight'] = rand_tensor((int(seq_len), int(k)))

# Merge new weights into state_dict
state_dict.update(new_weights)

# Save as raw state_dict
os.makedirs(os.path.dirname(new_ckpt_path), exist_ok=True)
torch.save(state_dict, new_ckpt_path)
print(f"New Linformer checkpoint saved to {new_ckpt_path}")
print("New Linformer weights:")
for k in new_weights:
    print(f"{k}: {new_weights[k].shape}")
