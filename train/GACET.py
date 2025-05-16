"""
Graph‑Aware Cross‑domain EEG Transformer (GACET)

"""
from typing import List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import reset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from timm.models.layers import DropPath
from .utils import ChannelPositionManager

def _mlp(in_dim: int, hidden: int, out_dim: int, *, dropout: float) -> nn.Sequential:
	"""Two‑layer MLP with GELU activation and dropout."""
	return nn.Sequential(
		nn.Linear(in_dim, hidden),
		nn.GELU(),
		nn.Dropout(dropout),
		nn.Linear(hidden, out_dim),
	)


def _mha(embed_dim: int, num_heads: int, *, dropout: float) -> nn.MultiheadAttention:
	"""PyTorch Multi‑Head Attention with batch‑first convention."""
	return nn.MultiheadAttention(
		embed_dim=embed_dim,
		num_heads=num_heads,
		dropout=dropout,
		batch_first=True,
	)


def _drop_paths(num_layers: int, p: float) -> nn.ModuleList:
	"""Generate a stochastic‑depth schedule from 0 → *p* across *num_layers*."""
	rates = torch.linspace(0.0, p, steps=num_layers, dtype=torch.float32).tolist()
	return nn.ModuleList(
		DropPath(drop_prob=rate) if rate > 0 else nn.Identity() for rate in rates
	)


class PositionalEncoder(nn.Module):
	def __init__(self, max_seq_len: int, embedding_dim: int):
		super().__init__()
		self.max_seq_len = max_seq_len
		self.embedding_dim = embedding_dim
		self.position_embeddings = nn.Embedding(self.max_seq_len, self.embedding_dim)
	
	def _apply_positional_encoding_to_one(self, x: Tensor) -> Tensor:
		if x.ndim != 3 or x.size(2) != self.embedding_dim:
			raise ValueError(f"Input tensor shape error. Expected (B, S, {self.embedding_dim}), got {x.shape}")
		seq_len = x.size(1)
		if seq_len > self.max_seq_len:
			raise ValueError(f"Input seq_len ({seq_len}) > max_seq_len ({self.max_seq_len}).")
		
		position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (S)
		pos_encodings = self.position_embeddings(position_ids)  # (S, D)
		pos_encodings = pos_encodings.unsqueeze(0)  # (1, S, D)
		return x + pos_encodings
	
	def forward(self, *tensors: Tensor) -> Tuple[Tensor, ...]:
		if not tensors:
			return tuple()
		return tuple(self._apply_positional_encoding_to_one(x) for x in tensors)


class FeedForward(nn.Module):
	"""Position‑wise FFN **without** internal LayerNorm (handled by caller)."""
	
	def __init__(self, dim: int, dropout: float) -> None:
		super().__init__()
		self.ffn = _mlp(dim, dim * 4, dim, dropout=dropout)
	
	def forward(self, x: Tensor) -> Tensor:  # (B, S, D)
		return self.ffn(x)


class CrossFreqAxialAttention(nn.Module):
	"""Axial (frequency × channel) attention with post‑norm residual paths."""
	
	def __init__(self, num_channels: int, dropout: float):
		super().__init__()
		self.attn = nn.MultiheadAttention(
			embed_dim=num_channels,
			num_heads=2,
			dropout=dropout,
			batch_first=True,
		)
		self.norm1 = nn.LayerNorm(num_channels)
		self.ffn = nn.Sequential(
			nn.Linear(num_channels, num_channels),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(num_channels, num_channels),
		)
		self.norm2 = nn.LayerNorm(num_channels)
	
	def forward(self, x_sp: Tensor) -> Tensor:  # (B, S, N, F)
		bsz, seq, n_ch, n_f = x_sp.shape
		x = x_sp.permute(0, 1, 3, 2).reshape(bsz * seq, n_f, n_ch)  # (B*S, F, N)
		
		attn_out, _ = self.attn(x, x, x)
		x = self.norm1(x + attn_out)
		x = self.norm2(x + self.ffn(x))
		
		return x.view(bsz, seq, n_f, n_ch).permute(0, 1, 3, 2)  # (B, S, N, F)


class AdaptiveDualFusionBlock(nn.Module):
	"""Bidirectional cross‑modal interaction with gated fusion (post‑norm)."""
	
	def __init__(
			self,
			dim: int,
			*,
			heads: int,
			layers: int,
			dropout: float,
			drop_path: float,
	) -> None:
		super().__init__()
		self.layers = layers
		
		self.attn_ab = nn.ModuleList(_mha(dim, heads, dropout=dropout) for _ in range(layers))
		self.attn_ba = nn.ModuleList(_mha(dim, heads, dropout=dropout) for _ in range(layers))
		
		self.ffn_a = nn.ModuleList(FeedForward(dim, dropout) for _ in range(layers))
		self.ffn_b = nn.ModuleList(FeedForward(dim, dropout) for _ in range(layers))
		
		# Stochastic depth schedules
		self.dp_attn_a = _drop_paths(layers, drop_path)
		self.dp_attn_b = _drop_paths(layers, drop_path)
		self.dp_ffn_a = _drop_paths(layers, drop_path)
		self.dp_ffn_b = _drop_paths(layers, drop_path)
		
		# LayerNorms (post‑residual)
		self.norm_attn_a = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
		self.norm_attn_b = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
		self.norm_ffn_a = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
		self.norm_ffn_b = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
		
		# Gated fusion
		self.fusion_mlp = nn.Sequential(
			nn.Linear(dim * 2, dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(dim, dim * 2),
		)
		self.final_norm = nn.LayerNorm(dim)
	
	def forward(self, a: Tensor, b: Tensor) -> Tensor:  # each (B, S, D)
		for i in range(self.layers):
			# A attends to B
			attn_a, _ = self.attn_ab[i](a, b, b)
			a = self.norm_attn_a[i](a + self.dp_attn_a[i](attn_a))
			
			a_ffn = self.ffn_a[i](a)
			a = self.norm_ffn_a[i](a + self.dp_ffn_a[i](a_ffn))
			
			# B attends to A
			attn_b, _ = self.attn_ba[i](b, a, a)
			b = self.norm_attn_b[i](b + self.dp_attn_b[i](attn_b))
			
			b_ffn = self.ffn_b[i](b)
			b = self.norm_ffn_b[i](b + self.dp_ffn_b[i](b_ffn))
		
		# Gated fusion
		ctx = torch.cat([a, b], dim=-1)  # (B, S, 2D)
		gating = self.fusion_mlp(ctx).view(*ctx.shape[:2], 2, -1)  # (B, S, 2, D)
		w_a, w_b = torch.softmax(gating, dim=2).unbind(dim=2)
		fused = w_a * a + w_b * b
		
		return self.final_norm(fused)

class AxialAttentionFusion(nn.Module):
	"""Cross‑modal fusion followed by axial frequency×channel attention."""
	
	def __init__(
			self,
			*,
			embed_dim: int,
			heads: int,
			fusion_layers: int,
			dropout: float,
			drop_path: float,
			num_channels: int,
			freq_bands: int,
	) -> None:
		super().__init__()
		self.cross_domain = AdaptiveDualFusionBlock(
			dim=embed_dim,
			heads=heads,
			layers=fusion_layers,
			dropout=dropout,
			drop_path=drop_path,
		)
		self.cross_freq_axial = CrossFreqAxialAttention(
			num_channels=num_channels, dropout=dropout
		)
		self.num_channels = num_channels
		self.freq_bands = freq_bands
	
	def forward(self, a: Tensor, b: Tensor) -> Tensor:  # each (B, S, D)
		fused = self.cross_domain(a, b)  # (B, S, D)
		bsz, seq, _ = fused.shape
		x_sp = fused.view(bsz, seq, self.num_channels, self.freq_bands)
		x_sp = self.cross_freq_axial(x_sp)
		return x_sp.reshape(bsz, seq, -1)


class DynamicGraphModule(nn.Module):
	"""Dynamic graph construction and TransformerConv aggregation."""
	
	def __init__(
			self,
			pos: torch.Tensor,  # (N, coord_dim)
			base_dim: int,
			feat_dim: int,
			*,
			latent_dim: int,
			heads: int,
			dropout: float,
			initial_tau: float,
			min_tau: float = 0.01,
			eps: float = 1e-6,
	) -> None:
		super().__init__()
		n_nodes, coord_dim = pos.shape
		self.n_nodes = n_nodes
		self.eps = eps
		
		# Position preprocessing
		pos_centered = pos - pos.mean(0, keepdim=True)
		pos_normed = F.normalize(pos_centered, p=2, dim=1)
		self.register_buffer("pos_tensor", pos_normed)
		self.pos_proj = nn.Linear(coord_dim, latent_dim, bias=False)
		
		# Delta MLP
		self.delta_mlp = nn.Sequential(
			nn.Linear(base_dim, base_dim * 2),
			nn.GELU(),
			nn.Linear(base_dim * 2, n_nodes * latent_dim),
		)
		
		# Learnable temperature
		self.log_tau = nn.Parameter(torch.log(torch.tensor(float(initial_tau))))
		self.min_tau = min_tau
		
		# Learnable gate
		self.logit_theta = nn.Parameter(torch.tensor(0.0))
		self.logit_alpha = nn.Parameter(torch.tensor(-2.0))
		
		# Distance weight
		self.raw_beta = nn.Parameter(torch.tensor(1.0))
		
		# Graph convolution
		self.graph_conv = TransformerConv(
			in_channels=feat_dim,
			out_channels=feat_dim,
			heads=heads,
			edge_dim=1,
			concat=False,
			dropout=dropout,
		)
		self.norm = nn.LayerNorm(feat_dim)
		self.norm2 = nn.LayerNorm(base_dim)
	
	def _current_tau(self) -> torch.Tensor:
		return torch.exp(self.log_tau).clamp(min=self.min_tau)
	
	def forward(self, pooled: Tensor, node_feats: Tensor) -> Tensor:  # pooled (B, D), node_feats (B, S, N, F)
		bsz, seq, n_nodes, f_dim = node_feats.shape
		assert n_nodes == self.n_nodes, "Node count mismatch"
		
		tau = self._current_tau()
		pos_emb = self.pos_proj(self.pos_tensor)  # (N, L)
		delta = self.delta_mlp(pooled).view(bsz, n_nodes, -1)
		z = pos_emb.unsqueeze(0) + delta  # (B, N, L)
		
		# Pairwise distances in latent and geometric spaces
		latent_dist = torch.cdist(z, z)
		geom_dist = torch.cdist(
			pos_emb.unsqueeze(0).expand(bsz, -1, -1),
			pos_emb.unsqueeze(0).expand(bsz, -1, -1),
		)
		logits = -(latent_dist + torch.abs(self.raw_beta) * geom_dist)
		
		# Gumbel‑softmax sampling
		gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + self.eps) + self.eps)
		y = (logits + gumbel_noise) / (tau + self.eps)
		soft_adj = torch.softmax(y, dim=-1)
		
		# Learnable gate
		theta = torch.sigmoid(self.logit_theta)
		alpha = torch.sigmoid(self.logit_alpha) * 10.0
		gate_prob = torch.sigmoid(alpha * (soft_adj - theta))
		hard_adj = (gate_prob > 0.5).float()
		adj_off = hard_adj.detach() - gate_prob.detach() + gate_prob
		adj_sym = 0.5 * (adj_off + adj_off.transpose(-1, -2))
		
		# Preserve soft diagonal
		diag_soft = torch.diagonal(soft_adj, dim1=-2, dim2=-1)
		idx = torch.arange(n_nodes, device=adj_sym.device)
		adj_sym[:, idx, idx] = diag_soft
		adj = adj_sym
		
		# Normalise adjacency
		deg = adj.sum(-1, keepdim=True).clamp(min=self.eps)
		adj_norm = deg.pow(-0.5) * adj * deg.pow(-0.5).transpose(-1, -2)
		
		# Graph convolution over the sequence dimension
		x_norm = self.norm(node_feats)
		data_list = []
		for b in range(bsz):
			edge_index, edge_weight = dense_to_sparse(adj_norm[b])
			x_b = x_norm[b].reshape(seq * n_nodes, f_dim)
			offset = torch.arange(seq, device=edge_index.device).view(-1, 1, 1) * n_nodes
			edge_index_b = (edge_index.unsqueeze(0) + offset).permute(1, 0, 2).reshape(2, -1)
			edge_weight_b = edge_weight.repeat(seq).unsqueeze(-1)
			data_list.append(Data(x=x_b, edge_index=edge_index_b, edge_attr=edge_weight_b))
		batch_graph = Batch.from_data_list(data_list)
		
		out = self.graph_conv(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
		out = out.view(bsz, seq, n_nodes, f_dim)
		res = node_feats + out
		res = self.norm2(res.reshape(bsz, seq, -1))
		return res


class CLSEncoder(nn.Module):
	
	def __init__(
			self,
			embed_dim: int,
			num_heads_attn: int,
			num_layers_transformer: int,
			dropout_rate: float,
	):
		super().__init__()
		
		self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
		
		encoder_layers = []
		for _ in range(num_layers_transformer):
			layer = nn.TransformerEncoderLayer(
				d_model=embed_dim,
				nhead=num_heads_attn,
				dim_feedforward=embed_dim * 4,
				dropout=dropout_rate,
				activation='gelu',
				batch_first=True,
			)
			encoder_layers.append(layer)
		self.transformer_encoder = nn.Sequential(*encoder_layers)
		self.layer_norm = nn.LayerNorm(embed_dim)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size = x.size(0)
		cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		
		x = self.transformer_encoder(x)
		cls = x[:, 0]
		return cls


class ClassificationHead(nn.Module):
	"""Two‑layer MLP classifier with LayerNorm and GELU."""
	
	def __init__(
			self,
			embed_dim: int,
			num_classes: int,
			dropout: float,
	):
		super().__init__()
		
		num_hidden_layers = math.ceil((num_classes - 1) / 2)
		
		hidden_size = embed_dim // 2
		layers = []
		in_features = embed_dim
		
		for _ in range(num_hidden_layers):
			layers.append(nn.Linear(in_features, hidden_size))
			layers.append(nn.GELU())
			layers.append(nn.Dropout(dropout))
			in_features = hidden_size
			hidden_size = hidden_size // 2
		
		layers.append(nn.Linear(in_features, num_classes))
		
		self.fc = nn.Sequential(*layers)
	
	def forward(self, x: Tensor) -> Tensor:
		return self.fc(x)


# ---------------------------------------------------------------------------
# Main model – GACET (post‑norm)
# ---------------------------------------------------------------------------

class GACET(nn.Module):
	"""Graph‑Aware Cross‑domain EEG Transformer (post‑norm)."""
	
	def __init__(
			self,
			electrode_pos = ChannelPositionManager().get_positions(source='data_2'),
			*,
			seq_len: int = 4,
			freq_bands: int = 5,
			num_classes: int = 2,
			embed_dim: int = 310,
			heads: int = 5,
			fusion_layers: int = 4,
			hyper_layers: int = 1,
			dropout: float = 0.3,
			drop_path: float = 0.4,
			latent_dim: int = 8,
			tau: float = 0.8,
			device: str | torch.device | None = None,
	) -> None:
		super().__init__()
		
		self.freq_bands = freq_bands
		assert embed_dim % freq_bands == 0, "embed_dim must be divisible by freq_bands"
		self.num_channels = embed_dim // freq_bands
		
		self.input_embed = PositionalEncoder(seq_len, embed_dim)
		
		# Cross‑modal and freq‑channel fusion
		self.axial_fusion = AxialAttentionFusion(
			embed_dim=embed_dim,
			heads=heads,
			fusion_layers=fusion_layers,
			dropout=dropout,
			drop_path=drop_path,
			num_channels=self.num_channels,
			freq_bands=freq_bands,
		)
		
		# Dynamic graph
		pos = torch.as_tensor(electrode_pos, dtype=torch.float32, device=device)
		self.dynamic_graph = DynamicGraphModule(
			pos=pos,
			base_dim=embed_dim,
			feat_dim=freq_bands,
			latent_dim=latent_dim,
			heads=heads,
			dropout=dropout,
			initial_tau=tau,
		)
		
		self.cls = CLSEncoder(embed_dim, heads, hyper_layers, dropout)
		self.classifier = ClassificationHead(embed_dim, num_classes, dropout)
		
		self.apply(self._init_weights)
	
	# ------------------------------------------------------------------
	@staticmethod
	def _init_weights(m: nn.Module) -> None:
		if isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight)
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, nn.Embedding):
			nn.init.normal_(m.weight, 0.0, 0.02)
		elif isinstance(m, nn.LayerNorm):
			nn.init.ones_(m.weight)
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, nn.MultiheadAttention):
			if hasattr(m, "in_proj_weight"):
				nn.init.xavier_uniform_(m.in_proj_weight, gain=1 / math.sqrt(2))
			if hasattr(m, "in_proj_bias") and m.in_proj_bias is not None:
				nn.init.zeros_(m.in_proj_bias)
			nn.init.xavier_uniform_(m.out_proj.weight)
			if m.out_proj.bias is not None:
				nn.init.zeros_(m.out_proj.bias)
		elif isinstance(m, TransformerConv):
			reset(m)
	
	# ------------------------------------------------------------------
	# Forward helpers
	# ------------------------------------------------------------------
	def _forward_features(self, eeg: List[Tensor]) -> Tuple[Tensor, Tensor | None]:
		a, b = self.input_embed(*eeg)  # (B, S, D)
		fused = self.axial_fusion(a, b)  # (B, S, D)
		bsz, seq, _ = fused.shape
		x_sp = fused.view(bsz, seq, self.num_channels, self.freq_bands)  # (B, S, N, F)
		pooled = x_sp.reshape(bsz, seq, -1).mean(1)  # (B, D)
		out_graph = self.dynamic_graph(pooled, x_sp)  # (B, S, N, F)
		cls_feat = self.cls(out_graph)
		return cls_feat
	
	def forward(self, eeg: List[Tensor]) -> Tensor | Tuple[Tensor, Tensor]:
		cls_feat = self._forward_features(eeg)
		logits = self.classifier(cls_feat)
		return logits
