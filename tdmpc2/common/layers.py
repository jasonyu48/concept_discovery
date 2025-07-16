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


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0., layernorm: bool = True):
	"""
	Basic building block of TD-MPC2.

	Args:
		in_dim (int): Input feature dimension.
		mlp_dims (int | list[int]): Hidden layer dimensions.
		out_dim (int): Output feature dimension.
		act (nn.Module | None): Optional activation applied **after** the final layer.
		dropout (float): Dropout probability applied *only* to the first hidden layer
			(retaining the behaviour of the original implementation).
		layernorm (bool): When ``True`` every hidden layer is a ``NormedLinear``
			(LayerNorm + Mish). When ``False`` we build plain ``nn.Linear`` layers
			without any LayerNorm.  The default preserves existing behaviour.
	"""

	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]

	dims = [in_dim] + mlp_dims + [out_dim]
	layers_out: list[nn.Module] = []

	for i in range(len(dims) - 2):
		in_d, out_d = dims[i], dims[i + 1]

		if layernorm:
			# Original behaviour: NormedLinear (Linear → Dropout → LayerNorm → Mish)
			layers_out.append(NormedLinear(in_d, out_d, dropout=dropout * (i == 0)))
		else:
			# Plain Linear without LayerNorm
			mods = [nn.Linear(in_d, out_d)]
			if dropout and i == 0:
				mods.append(nn.Dropout(dropout, inplace=False))
			# Mirror original default activation (Mish) for hidden layers
			mods.append(nn.Mish(inplace=False))
			layers_out.append(nn.Sequential(*mods))

	# ---- Final layer ----
	last_in, last_out = dims[-2], dims[-1]
	if act is not None:
		if layernorm:
			layers_out.append(NormedLinear(last_in, last_out, act=act))
		else:
			layers_out.append(nn.Sequential(nn.Linear(last_in, last_out), act))
	else:
		layers_out.append(nn.Linear(last_in, last_out))

	return nn.Sequential(*layers_out)


# -----------------------------------------------
# CNN encoder helper
# -----------------------------------------------


def conv(in_shape, num_channels, latent_dim, act=None, *, shiftaug: bool = True):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.

	Args:
		in_shape (Tuple[int]): (C, H, W) input shape; H, W must be 64.
		num_channels (int): Base channel width (default 32).
		latent_dim (int): Expected latent dimension (must equal ``num_channels*4*4``).
		act (nn.Module | None): Optional final activation (e.g. SimNorm).
		shiftaug (bool): Apply random shift augmentation if ``True`` (default).
	"""
	assert in_shape[-1] == 64  # assumes rgb observations to be 64x64

	# Calculate flattened conv output size: num_channels * 4 * 4 for 64x64 input
	conv_output_size = num_channels * 4 * 4

	# The conv stack outputs a fixed 512-D vector (for default num_channels=32).
	# This is **independent** of cfg.latent_dim, therefore we explicitly assert
	# that the configuration matches this architectural constant to avoid
	# silent shape mismatches downstream.
	assert latent_dim == conv_output_size, (
		f"CNN encoder produces {conv_output_size}-D latents, but cfg.latent_dim="
		f"{latent_dim}. Either set latent_dim={conv_output_size} or switch to a"
		" different encoder_arch (e.g. 'vit' or 'mlp')."
	)

	layers: list[nn.Module] = []
	# Optional spatial shift augmentation
	if shiftaug:
		layers.append(ShiftAug())
	# Pixel normalisation
	layers.append(PixelPreprocess())
	# Convolutional stem
	layers += [
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten(),
	]
	if act is not None:
		layers.append(act)
	return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# New RGB MLP encoder (flattened pixels → latent)
# -----------------------------------------------------------------------------


class RGBMLPEncoder(nn.Module):
	"""MLP encoder operating on flattened 64×64 RGB observations."""

	def __init__(self, cfg):
		super().__init__()
		in_dim = cfg.obs_shape["rgb"][0] * 64 * 64  # C⋅H⋅W
		# Build hidden layers following state-encoder convention
		n_layers = max(cfg.num_enc_layers - 1, 1)
		hidden_dims = [cfg.enc_dim] * n_layers
		# Optional random shift augmentation (enabled by default)
		use_shift = getattr(cfg, "enc_shiftaug", True)
		self.shift = ShiftAug() if use_shift else nn.Identity()
		self.pre = PixelPreprocess()

		# Whether to include LayerNorm within the encoder MLP is controlled by
		# cfg.enc_layernorm (default: True).
		use_ln = getattr(cfg, "enc_layernorm", True)
		self.mlp = mlp(in_dim, hidden_dims, cfg.latent_dim, layernorm=use_ln)

	def forward(self, x):
		# x: (B, C, 64, 64)
		x = self.shift(x.float())
		x = self.pre(x)
		x = x.flatten(1)
		return self.mlp(x)


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
        # Optional shift augmentation controlled by cfg.enc_shiftaug (default True)
        use_shift = getattr(cfg, "enc_shiftaug", True)
        self.shift_aug = ShiftAug() if use_shift else nn.Identity()
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


# -----------------------------------------------------------------------------
# Deep ResNet-style encoder (64×64 RGB → latent)
# -----------------------------------------------------------------------------


class _BasicBlock(nn.Module):
	"""Minimal pre-activation ResNet basic block (Conv→ReLU→Conv + skip)."""

	def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
		super().__init__()
		self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=False)
		self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
		self.downsample = None
		if stride != 1 or in_ch != out_ch:
			self.downsample = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
		identity = x
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		if self.downsample is not None:
			identity = self.downsample(identity)
		x = x + identity
		return self.relu(x)


class ResNetEncoder(nn.Module):
	"""Configurable deep ResNet-style encoder.

	Key features
	-------------
	• Optional `ShiftAug` & pixel normalisation.
	• Depth controlled by `cfg.resnet_blocks` (number of *blocks per stage*, default 2 ⇒ ResNet-18 layout).
	• Kaiming (He) initialisation applied **only** to this encoder’s convolutional layers to avoid
	  interfering with the global `init.weight_init` that WorldModel applies to linear layers.
	"""

	def __init__(self, cfg):
		super().__init__()
		C0 = cfg.num_channels  # base width (e.g. 32)
		use_shift = getattr(cfg, "enc_shiftaug", True)
		self.shift = ShiftAug() if use_shift else nn.Identity()
		self.pre = PixelPreprocess()

		# ----- Stem -----
		self.stem = nn.Sequential(
			nn.Conv2d(cfg.obs_shape["rgb"][0], C0, kernel_size=7, stride=2, padding=3, bias=False),
			nn.ReLU(inplace=False),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)

		# ----- Residual stages -----
		blocks_per_stage = int(getattr(cfg, "resnet_blocks", 2))
		channels = [C0, 2 * C0, 4 * C0, 8 * C0]
		strides = [1, 2, 2, 2]  # first stage keeps spatial size
		stages = []
		in_ch = C0
		for out_ch, st in zip(channels, strides):
			blk = []
			# First block may downsample
			blk.append(_BasicBlock(in_ch, out_ch, stride=st))
			in_ch = out_ch
			# Additional blocks with stride=1
			for _ in range(blocks_per_stage - 1):
				blk.append(_BasicBlock(in_ch, in_ch, stride=1))
			stages.append(nn.Sequential(*blk))
		self.stages = nn.Sequential(*stages)

		# ----- Head -----
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.proj = nn.Linear(in_ch, cfg.latent_dim)

		# ----- Kaiming initialisation for conv layers -----
		self.apply(self._kaiming_init)

	@staticmethod
	def _kaiming_init(m):
		"""Kaiming normal initialisation for Conv/Linear layers with ReLU non-linearity."""
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
		x = self.shift(x.float())
		x = self.pre(x)
		x = self.stem(x)
		x = self.stages(x)
		x = self.avgpool(x).flatten(1)
		return self.proj(x)


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
			elif encoder_arch == 'mlp':
				# New MLP encoder path
				mlp_enc = RGBMLPEncoder(cfg)
				if cfg.simnorm:
					mlp_enc = nn.Sequential(mlp_enc, SimNorm(cfg))
				out[k] = mlp_enc
			elif encoder_arch == 'resnet':
				res_enc = ResNetEncoder(cfg)
				if cfg.simnorm:
					res_enc = nn.Sequential(res_enc, SimNorm(cfg))
				out[k] = res_enc
			elif encoder_arch == 'cnn':  # default CNN encoder path
				shift = getattr(cfg, "enc_shiftaug", True)
				if cfg.simnorm:
					out[k] = conv(cfg.obs_shape[k], cfg.num_channels, cfg.latent_dim, act=SimNorm(cfg), shiftaug=shift)
				else:
					out[k] = conv(cfg.obs_shape[k], cfg.num_channels, cfg.latent_dim, shiftaug=shift)
			else:
				raise ValueError(f"Unsupported encoder_arch '{encoder_arch}'.")
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
        expected_flat = self.h_dim * 4 * 4  # expected latent size (e.g. 512)

        # If latent_dim differs, insert a projection layer so decoder can ingest
        # any latent size without breaking the existing conv-transpose stack.
        if cfg.latent_dim != expected_flat:
            self.project = nn.Linear(cfg.latent_dim, expected_flat)
        else:
            self.project = nn.Identity()

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
        B, _ = z.shape
        z_proj = self.project(z)
        x = z_proj.view(B, self.h_dim, 4, 4)
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

# === i-ResNet Transition Model (Lipschitz-controlled) ===
# Inspired by https://arxiv.org/abs/1810.12933. Provides an invertible residual
# dynamics model z' = z + g([z,a]) with spectral normalisation to guarantee
# Lipschitz constant ≤ C.

def SN(layer: nn.Module, coeff: float) -> nn.Module:
    """Applies spectral normalization and rescales the *output* by ``coeff``.

    PyTorch's ``spectral_norm`` constrains the **spectral norm** (largest
    singular value) of the weight matrix to 1.  To achieve a minimum singular
    value ≥ 1 / C (hence overall Lipschitz constant ≤ C) we simply multiply the
    *weights* by ``coeff = 1 − 1/C``.  Rather than manipulating the parameters
    directly we scale the *output* of the layer which is algebraically
    equivalent and avoids interfering with PyTorch's parameterization.
    """

    base = torch.nn.utils.spectral_norm(layer)

    class _SNScaled(nn.Module):
        def __init__(self, mod: nn.Module, scale: float):
            super().__init__()
            self.mod = mod
            self.scale = scale

        def forward(self, x):  # noqa: D401
            return self.mod(x) * self.scale

        def extra_repr(self):  # noqa: D401
            return f"scale={self.scale}"

    return _SNScaled(base, coeff)


class IResNetTransition(nn.Module):
    """Invertible residual transition model with spectral-norm control.

    We implement
        z' = z + g([z, task_emb, a])
    with g parameterised by a Spectral-Norm MLP whose hidden layer widths and
    depth are determined by ``mlp_dims`` (taken from ``cfg.mlp_dim``).  When
    ``cfg.mlp_dim`` is an ``int`` we create a **single** hidden layer of that
    width; when it is a *list/tuple* we create one hidden layer per element.

    The *Lipschitz* constant of g is bounded by ``coeff = 1 − 1/C`` where
    ``C > 1`` is asserted, guaranteeing the desired contraction inequality
    ‖z₁ − z₂‖ ≤ C‖z₁' − z₂'‖ for every action and task.
    """

    def __init__(self, in_dim: int, z_dim: int, mlp_dims, C: float = 2.0, simple: bool = False):
        super().__init__()
        assert C > 1.0, "Lipschitz constant C must be > 1"

        coeff = 1.0 - 1.0 / C  # σ_min ≥ 1/C

        if simple:
            # Single spectral-norm linear layer
            self.g = SN(nn.Linear(in_dim, z_dim), coeff)
        else:
            # Handle flexible mlp_dims specification (int or list)
            if isinstance(mlp_dims, int):
                hidden_dims = [mlp_dims]
            else:
                hidden_dims = list(mlp_dims)

            dims = [in_dim] + hidden_dims + [z_dim]
            layers_sn = []
            for i in range(len(dims) - 1):
                layers_sn.append(SN(nn.Linear(dims[i], dims[i + 1]), coeff))
                if i < len(dims) - 2:  # no activation after final layer
                    layers_sn.append(nn.LeakyReLU(0.1, inplace=True))

            self.g = nn.Sequential(*layers_sn)
        self.z_dim = z_dim

    def forward(self, zcat: torch.Tensor) -> torch.Tensor:
        """Expect *concatenated* input (z, task_emb?, a)."""
        z = zcat[..., : self.z_dim]
        return z + self.g(zcat)

    def __repr__(self):
        """Rich representation showing the internal MLP like `nn.Sequential`.

        Matches the style used in WorldModel’s `__repr__` so that the transition
        section prints full layer details (comparable to the Reward MLP output).
        """
        inner_repr = repr(self.g)
        # Indent inner block by two spaces for nicer formatting alongside other modules
        indented = "  " + inner_repr.replace("\n", "\n  ")
        return f"IResNetTransition(\n{indented}\n)"
