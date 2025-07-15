import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1])) # the last layer has layernorm????
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, latent_dim, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	
	# Calculate flattened conv output size: num_channels * 4 * 4 for 64x64 input
	conv_output_size = num_channels * 4 * 4
	
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten(),
		# nn.Linear(conv_output_size, latent_dim)  # Project to latent_dim
	]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder that contains **no convolutional layers**.
    It operates on 64×64 RGB observations (stacked frames allowed, e.g. C=9)
    by:
        1. Patchifying the image via `nn.Unfold` (a view, not conv)
        2. Linear projection of flattened patches to `latent_dim`
        3. Adding learnable positional embeddings
        4. Passing through a stack of `TransformerEncoderLayer`s
        5. Mean-pooling the token dimension to obtain a single latent vector
    The design mirrors the conv encoder output dimension so the rest of the
    codebase remains unchanged.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # --- configuration ----
        self.patch_size = getattr(cfg, "patch_size", 8)  # 8×8 default
        C = cfg.obs_shape['rgb'][0]                      # 3 or 9
        D = cfg.latent_dim                               # e.g. 512
        num_patches = (64 // self.patch_size) ** 2       # 64×64 input guaranteed

        # --- modules ----
        self.shift_aug = ShiftAug()
        self.preprocess = PixelPreprocess()
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_embed = nn.Linear(C * self.patch_size * self.patch_size, D)
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, D))

        # Transformer stack
        depth = getattr(cfg, "vit_depth", 2)
        nhead = getattr(cfg, "vit_nhead", 2)
        mlp_ratio = getattr(cfg, "vit_mlp_ratio", 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=nhead,
            dim_feedforward=D * mlp_ratio,
            dropout=0.0,
            activation=nn.Mish(inplace=False),
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        """Expect x of shape (B, C, 64, 64). Returns (B, latent_dim)."""
        # Data augmentation & normalisation to match conv branch behaviour
        x = self.shift_aug(x.float())
        x = self.preprocess(x)

        # (B, C, H, W) -> (B, N_patches, patch_dim)
        patches = self.unfold(x).transpose(1, 2)
        tokens = self.patch_embed(patches)  # (B, N, D)
        tokens = tokens + self.pos_embed
        y = self.encoder(tokens)            # (B, N, D)
        z = self.norm(y.mean(dim=1))        # mean-pool -> (B, D)
        return z


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			if cfg.simnorm:
				out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
			else:
				out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim)
		elif k == 'rgb':
			encoder_arch = getattr(cfg, 'encoder_arch', 'cnn')
			if encoder_arch == 'vit':
				# Pure transformer encoder path
				vit_enc = ViTEncoder(cfg)
				if cfg.simnorm:
					vit_enc = nn.Sequential(vit_enc, SimNorm(cfg))
				out[k] = vit_enc
			else:  # default CNN encoder path
				if cfg.simnorm:
					out[k] = conv(cfg.obs_shape[k], cfg.num_channels, cfg.latent_dim, act=SimNorm(cfg))
				else:
					out[k] = conv(cfg.obs_shape[k], cfg.num_channels, cfg.latent_dim)
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


###adding the deocoder-Siyi Chen 2025-7-3#####
def dec(cfg):
    """Return a decoder that mirrors the encoder."""
    if cfg.obs == "rgb":
        return CNNDecoder(cfg)
    elif cfg.obs == "state":
        return MLPDecoder(cfg)          
    else:
        raise NotImplementedError(f"No decoder for obs={cfg.obs}")

# ===== 1.2 CNNDecoder（64×64 RGB → z inverse process） =============================
class CNNDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h_dim = cfg.num_channels #32
        # This allows it to reconstruct the full frame stack (e.g., 9 channels).
        output_channels = cfg.obs_shape['rgb'][0]
        self.net = nn.Sequential(
                    nn.ConvTranspose2d(self.h_dim, self.h_dim,
                                       3, stride=1, padding=1),  nn.ReLU(),        # 4×4
                    nn.ConvTranspose2d(self.h_dim, self.h_dim,
                                       3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 8×8
                    nn.ConvTranspose2d(self.h_dim, self.h_dim,
                                       5, stride=2, padding=2, output_padding=1), nn.ReLU(),  # 16×16
                    nn.ConvTranspose2d(self.h_dim, self.h_dim,
                                       5, stride=2, padding=2, output_padding=1), nn.ReLU(),  # 32×32
                    nn.ConvTranspose2d(self.h_dim, output_channels,
                                       4, stride=2, padding=1),                         # 64×64
                    nn.Tanh()   # [-1,1]
                )
    
    def forward(self, z):
        # z shape: (B, 32*4*4) = (B, 512)
        B, D = z.shape
        assert D == self.h_dim * 4 * 4, f"Expect {self.h_dim*4*4}, got {D}"
        x = z.view(B, self.h_dim, 4, 4)
        return self.net(x)

# =====MLPDecoder ============================================
class MLPDecoder(nn.Module):
    def __init__(self, cfg):
        raise NotImplementedError("State‑only tasks has not enbled Decoder")
	
######end of decoder part########



def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict
