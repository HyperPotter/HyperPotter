import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import math
from hgnn_memory_utils import ProtoMemoryManager


def sinusoidal_pos_encoding(num_positions: int, dim: int, device, dtype=torch.float32):
    """Classic extrapolatable sin/cos positional encoding (length-adaptive)."""
    if num_positions <= 0:
        return torch.zeros(1, 0, dim, device=device, dtype=dtype)
    position = torch.arange(0, num_positions, device=device, dtype=dtype).unsqueeze(1)  # (N,1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-(math.log(10000.0) / dim)))
    pe = torch.zeros(num_positions, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, N, dim)


def normalized_time_coords(num_positions: int, device, dtype=torch.float32):
    """Normalized time coordinates t/T within [0, 1] shaped as (1, N, 1)."""
    if num_positions <= 0:
        return torch.zeros(1, 0, 1, device=device, dtype=dtype)
    t = torch.linspace(0, 1, steps=num_positions, device=device, dtype=dtype).unsqueeze(-1)
    return t.unsqueeze(0)  # (1,N,1)


class FuzzyCMeansModule(nn.Module):
    """
    Differentiable Fuzzy C-Means clustering module used to refresh node embeddings.
    """
    def __init__(self, n_clusters, fuzziness=2.0, max_iter=10, eps=1e-6, init_method='kmeans++'):
        super().__init__()
        self.n_clusters = n_clusters
        self.m = fuzziness
        self.max_iter = max_iter
        self.eps = eps
        self.init_method = init_method
        self.external_init_centroids = None  # (B,K,C) or (K,C) supplied for external warm starts

        
    def _init_centroids_kmeans_plus(self, X):
        """
        K-means++ initialization strategy.
        """
        B, N, C = X.shape
        device = X.device
        
        centroids = torch.zeros(B, self.n_clusters, C, device=device, dtype=X.dtype)
        
        for b in range(B):
            x_b = X[b]  # (N, C)
            
            # Randomly select the first centroid
            first_idx = torch.randint(0, N, (1,), device=device)
            centroids[b, 0] = x_b[first_idx]
            for i in range(1, min(self.n_clusters, N)):
                # Compute the minimum distance to the already chosen centroids
                dists = torch.cdist(x_b.unsqueeze(0), centroids[b, :i].unsqueeze(0)).squeeze(0)
                min_dists = torch.min(dists, dim=1)[0]
                # Add a small constant to avoid all-zero probabilities
                min_dists = min_dists + self.eps
                # Compute sampling probabilities
                prob_sum = torch.sum(min_dists)
                if prob_sum <= self.eps:
                    probs = torch.ones_like(min_dists) / N
                else:
                    probs = min_dists / prob_sum
                # Ensure the probabilities sum to 1
                probs = probs / torch.sum(probs)
                # Select the next centroid
                if torch.sum(probs) > 0 and not torch.any(torch.isnan(probs)):
                    next_idx = torch.multinomial(probs, 1)
                else:
                    next_idx = torch.randint(0, N, (1,), device=device)
                centroids[b, i] = x_b[next_idx]
        return centroids

    def _init_centroids_random(self, X):
        """
        Random initialization.
        """
        B, N, C = X.shape
        device = X.device
        
        effective_clusters = min(self.n_clusters, N)
        centroids = torch.zeros(B, effective_clusters, C, device=device, dtype=X.dtype)
        
        for b in range(B):
            if N >= effective_clusters:
                idx = torch.randperm(N, device=device)[:effective_clusters]
                centroids[b] = X[b, idx]
            else:
                idx = torch.randint(0, N, (effective_clusters,), device=device)
                centroids[b] = X[b, idx]
        
        return centroids
    
    def _compute_membership_efficient(self, X, centroids):
        """
        Efficiently compute the membership matrix.
        """
        B, N, C = X.shape
        P = centroids.size(1)
        
        # Compute the distance matrix (B, N, P)
        dist = torch.cdist(X, centroids, p=2) + self.eps
        # Compute memberships: u_ij = 1 / sum_k (d_ij / d_ik)^(2/(m-1))
        power = 2.0 / (self.m - 1.0)
        inv_dist = 1.0 / (dist.pow(power) + self.eps)
        membership = inv_dist / (torch.sum(inv_dist, dim=-1, keepdim=True) + self.eps)
        return membership

    def _update_centroids(self, X, membership):
        """
        Update cluster centroids.
        """
        # Compute weighted memberships
        weighted_membership = membership.pow(self.m)
        # Compute new centroids
        numerator = torch.bmm(weighted_membership.transpose(1, 2), X)
        denominator = torch.sum(weighted_membership, dim=1, keepdim=True).transpose(1, 2)
        centroids = numerator / (denominator + self.eps)
        return centroids

    def forward(self, X):
        """
        Run Fuzzy C-Means clustering and return refreshed node embeddings.
        """
        B, N, C = X.shape
        device = X.device
        
        # Standardize data to improve numerical stability
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X_std = torch.std(X, dim=1, keepdim=True) + self.eps
        X_normalized = (X - X_mean) / X_std

        # Ensure the cluster count does not exceed the sample size
        effective_clusters = min(self.n_clusters, N)

        # Use externally provided centroids if available
        if (self.external_init_centroids is not None):
            if self.external_init_centroids.dim() == 2:
                centroids = self.external_init_centroids.unsqueeze(0).expand(B, -1, -1).contiguous()
            else:
                centroids = self.external_init_centroids
            # Map external centroids into the normalized space
            centroids = (centroids - X_mean) / X_std
            # Truncate K to avoid overflow
            K_eff = min(centroids.size(1), effective_clusters)
            centroids = centroids[:, :K_eff, :]
        else:
            # Keep the original kmeans++ / random initialization logic
            if self.init_method == 'kmeans++' and N >= effective_clusters:
                centroids = self._init_centroids_kmeans_plus(X_normalized)
            else:
                centroids = self._init_centroids_random(X_normalized)

        # Iteratively update centroids
        prev_centroids = centroids.clone()
        for iter_idx in range(self.max_iter):
            try:
                # Compute memberships
                membership = self._compute_membership_efficient(X_normalized, centroids)
                # Update centroids
                new_centroids = self._update_centroids(X_normalized, membership)
                # Check convergence
                if iter_idx > 0:
                    center_shift = torch.norm(new_centroids - centroids, dim=-1).mean()
                    if center_shift < self.eps:
                        break
                prev_centroids = centroids.clone()
                centroids = new_centroids
            except Exception as e:
                print(f"FCM iteration {iter_idx} failed: {e}")
                centroids = prev_centroids
                break

        # Final membership computation
        try:
            membership = self._compute_membership_efficient(X_normalized, centroids)
        except Exception:
            membership = torch.ones(B, N, effective_clusters, device=device) / effective_clusters

        # Produce hard cluster assignments
        cluster_assignments = torch.argmax(membership, dim=-1)

        # De-standardize centroids back to the original space
        centroids_original = centroids * X_std + X_mean

        # Use FCM outputs to refresh node embeddings
        # Method 1: membership-weighted update
        weighted_centroids = torch.bmm(membership, centroids_original)
        # Method 2: blend original features with clustering cues
        alpha = 0.9
        updated_embeddings = alpha * X + (1 - alpha) * weighted_centroids

        # Note: external_init_centroids is cleared elsewhere by ProtoMemoryManager
        return updated_embeddings, centroids_original, membership, cluster_assignments

############################
## FOR fine-tuned SSL MODEL
############################


class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = '/path/to/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return
        

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
        
        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb




class HypergraphConvLayer(nn.Module):
    """
    Hypergraph Attention layer with FuzzyCMeans-based node embedding updates to replace the original graph attention layer.
    """
    def __init__(self, in_dim, out_dim, dropout=0.2, use_attention=True, n_clusters=8, **kwargs):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_attention = use_attention
        self.n_clusters = n_clusters
        # FuzzyCMeans module used for node embedding updates
        self.fcm_module = FuzzyCMeansModule(
            n_clusters=n_clusters,
            fuzziness=2.0,
            max_iter=5,
            eps=1e-6,
            init_method='kmeans++'
        )
        # Node feature transformation
        self.node_proj = nn.Linear(in_dim, out_dim)
        # Cluster-enhanced feature projection
        self.cluster_proj = nn.Linear(in_dim, out_dim)
        # Attention mechanism - simplified to avoid dimensional issues
        self.use_attention = use_attention
        if self.use_attention:
            # Use a lightweight dot-product attention instead of multi-head attention
            self.attention_proj = nn.Linear(out_dim, out_dim)
            self.attention_weight = nn.Parameter(torch.randn(out_dim, 1))
        # Feature fusion gate
        self.fusion_gate = nn.Linear(out_dim * 2, out_dim)
        # Normalization and activation
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SELU(inplace=True)
        self.proto_manager = None  # Injected by the model inside __init__
        self.proto_key = None      # Unique key per layer (e.g., 'HGNN_layer_S')
        # Temperature parameter
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        """
        x: (batch_size, num_nodes, in_dim)
        """
        batch_size, num_nodes, _ = x.size()

        # ---- (A) FCM warm start: only use the global memory during evaluation ----
        if (self.proto_manager is not None) and ((not self.training) or getattr(self.proto_manager, 'warm_on_train', False)):
            self.proto_manager.maybe_warm_start(self.proto_key, self.fcm_module, x, self.n_clusters)

        # Step 1: update node embeddings with FuzzyCMeans
        updated_embeddings, centroids, membership, cluster_assignments = self.fcm_module(x)

        # ---- (C) Training phase: update the prototype banks with labels ----
        if (self.proto_manager is not None) and self.training:
            m = getattr(self.fcm_module, "m", 2.0)
            if self.proto_manager.use_slot_alignment:
                self.proto_manager.update_slot(self.proto_key, x, centroids, membership, m)
            else:
                self.proto_manager.update_raw(self.proto_key, x, centroids, membership, m)

        # Step 2: node feature transformation (uses the refreshed embeddings)
        node_feat = self.node_proj(updated_embeddings)

        # Step 3: cluster-enhanced features
        cluster_feat = self.cluster_proj(updated_embeddings)

        # Step 4: derive amplification operator
        operator = self._amplification_operator(node_feat, membership)

        # Step 5: detection amplification
        conv_output = self._detection_amplification(node_feat, operator)

        # Step 6: fuse hypergraph output with cluster-enhanced features
        fused_features = self._fuse_features(conv_output, cluster_feat)

        # Step 7: residual connection and normalization
        if self.in_dim == self.out_dim:
            output = fused_features + self.node_proj(x)  # Use the original x for the residual connection
        else:
            output = fused_features

        output = self.norm(output)
        output = self.activation(output)

        return output

    def _amplification_operator(self, node_feat, membership):
        """
        An amplification operator that combines cluster membership and feature similarity.
        """
        batch_size, num_nodes, feat_dim = node_feat.size()

        # Method 1: use membership-vector similarity
        membership_norm = F.normalize(membership, p=2, dim=-1)
        cluster_similarity = torch.bmm(membership_norm, membership_norm.transpose(-2, -1))

        # Method 2: use feature similarity
        feat_similarity = torch.matmul(node_feat, node_feat.transpose(-2, -1)) / math.sqrt(feat_dim)

        # Fuse the two similarities
        alpha = 0.6  # Weight for cluster similarity
        combined_similarity = alpha * cluster_similarity + (1 - alpha) * feat_similarity
        combined_similarity = combined_similarity / self.temp

        # Use softmax to obtain hyperedge weights
        operator = F.softmax(combined_similarity, dim=-1)

        return operator


    def _fuse_features(self, conv_output, cluster_feat):
        """
        Fuse the hypergraph convolution output with the cluster-enhanced features.
        """
        # Concatenate features
        combined_feat = torch.cat([conv_output, cluster_feat], dim=-1)

        # Gated fusion
        fusion_weights = torch.sigmoid(self.fusion_gate(combined_feat))

        # Weighted fusion
        fused_output = fusion_weights * conv_output + (1 - fusion_weights) * cluster_feat

        return fused_output

    def _detection_amplification(self, node_feat, operator):
        """
        Perform detection amplification via optional attention.
        """
        # Step 1: compute relational evidence
        relational_evidence = torch.matmul(operator, node_feat)

        # Use a simplified attention mechanism to enhance features
        if self.use_attention:
            try:
                # Lightweight self-attention mechanism
                att_proj = self.attention_proj(relational_evidence)
                att_logits = torch.matmul(att_proj, self.attention_weight).squeeze(-1)
                # Apply temperature scaling to control softmax smoothness
                att_logits = att_logits / self.temp
                att_weights = F.softmax(att_logits, dim=-1).unsqueeze(-1)

                # Weighted features
                relational_evidence_att = relational_evidence * att_weights
                relational_evidence = relational_evidence + relational_evidence_att
            except RuntimeError as e:
                # If attention fails, skip the enhancement step
                print(f"Skip attention: {e}")
                pass

        # Step 2: propagate relational evidence back to nodes
        output = torch.matmul(operator.transpose(-2, -1), relational_evidence)

        output = self.dropout(output)

        return output

class HeterogeneousHGNNLayer(nn.Module):
    """
    Heterogeneous hypergraph neural network layer with integrated FuzzyCMeans to replace HtrgGraphAttentionLayer.
    """
    def __init__(self, in_dim, out_dim, dropout=0.2, n_clusters=8, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_clusters = n_clusters
        # temperature for attention/softmax scaling
        self.temp = 1.0
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

        # FuzzyCMeans module used for node embedding updates
        self.fcm_module = FuzzyCMeansModule(
            n_clusters=n_clusters,
            fuzziness=2.0,
            max_iter=5,
            eps=1e-6,
            init_method='kmeans++'
        )

        # Projection layers for different node types
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # Hypergraph convolution layer (propagates n_clusters)
        self.hgnn_conv = HypergraphConvLayer(in_dim, out_dim, dropout, n_clusters=n_clusters, **kwargs)

        # Master-node-related layers
        self.master_proj = nn.Linear(in_dim, out_dim)
        self.master_attention = nn.MultiheadAttention(out_dim, num_heads=4, dropout=dropout, batch_first=True)

        # Cluster-enhanced master-node update
        self.master_cluster_proj = nn.Linear(in_dim, out_dim)
        self.master_fusion_gate = nn.Linear(out_dim * 2, out_dim)

        # Normalization
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SELU(inplace=True)

        self.proto_manager = None  # Injected by the model inside __init__
        self.proto_key = None      # Unique key per layer (e.g., 'HGNN_layer_S')

    def forward(self, x1, x2, master=None):
        """
        x1: type1 nodes (batch_size, num_nodes1, in_dim)
        x2: type2 nodes (batch_size, num_nodes2, in_dim)
        master: master node (batch_size, 1, in_dim)
        """
        batch_size = x1.size(0)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        # Type-specific projections
        x1_proj = self.proj_type1(x1)
        x2_proj = self.proj_type2(x2)

        # Merge different node types
        x_combined = torch.cat([x1_proj, x2_proj], dim=1)

        # Derive master node if not provided
        if master is None:
            master = torch.mean(x_combined, dim=1, keepdim=True)

        # ---- (A) FCM warm start: only use the global memory during evaluation ----
        if (self.proto_manager is not None) and ((not self.training) or getattr(self.proto_manager, 'warm_on_train', False)):
            self.proto_manager.maybe_warm_start(self.proto_key, self.fcm_module, x_combined, self.n_clusters)

        # Step 1: refresh the merged node embeddings with FuzzyCMeans
        updated_combined, centroids, membership, cluster_assignments = self.fcm_module(x_combined)

        # ---- (C) Training phase: update the prototype banks with labels ----
        if (self.proto_manager is not None) and self.training:
            m = getattr(self.fcm_module, "m", 2.0)
            if self.proto_manager.use_slot_alignment:
                self.proto_manager.update_slot(self.proto_key, x_combined, centroids, membership, m)
            else:
                self.proto_manager.update_raw(self.proto_key, x_combined, centroids, membership, m)

        # Step 2: hypergraph convolution (using refreshed embeddings)
        x_conv = self.hgnn_conv(updated_combined)

        # Step 3: update the master node with clustering cues
        master_updated = self._update_master_with_clusters(x_conv, master, centroids, membership)

        # Step 4: apply normalization and activation
        x_conv = self.norm(x_conv)
        x_conv = self.activation(x_conv)

        # Step 5: split node types
        x1_out = x_conv.narrow(1, 0, num_type1)
        x2_out = x_conv.narrow(1, num_type1, num_type2)

        return x1_out, x2_out, master_updated

    def _update_master_with_clusters(self, x, master, centroids, membership):
        """
        Update the master node using clustering information.
        """
        # Method 1: conventional attention update
        master_proj = self.master_proj(master)

        # Use a simpler attention mechanism to avoid multi-head mismatches
        try:
            # Compute attention weights: master attends to all nodes
            attention_scores = torch.matmul(master_proj, x.transpose(-2, -1))
            # Apply temperature scaling to control softmax smoothness
            attention_scores = attention_scores / self.temp
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Apply attention weights
            attention_out = torch.matmul(attention_weights, x)
            traditional_update = master_proj + attention_out
        except Exception as e:
            print(f"Fallback to a simple global average: {e}")
            # Fallback to a simple global average
            global_avg = torch.mean(x, dim=1, keepdim=True)
            traditional_update = master_proj + global_avg

        # Method 2: cluster-centric update
        # Compute a global representation from cluster centroids
        cluster_global = torch.mean(centroids, dim=1, keepdim=True)
        cluster_update = self.master_cluster_proj(cluster_global)

        # Fuse both updates
        combined_features = torch.cat([traditional_update, cluster_update], dim=-1)
        fusion_weights = torch.sigmoid(self.master_fusion_gate(combined_features))

        master_updated = fusion_weights * traditional_update + (1 - fusion_weights) * cluster_update

        return master_updated

    def _update_master(self, x, master):
        """
        Update the master node (legacy version kept for backward compatibility).
        """
        # Use attention mechanism to update the master node
        master_proj = self.master_proj(master)
        
        # Compute attention weights
        attention_out, attention_weights = self.master_attention(master_proj, x, x)
        
        # Residual connection
        master_updated = master_proj + attention_out
        
        return master_updated


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.LayerNorm(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # directional edge for master node
        master = self._update_master(x, master)
        # projection
        x = self._project(x, att_map)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h




class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out


class Model(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 6.0, 6.0]


        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
            
        )

        # DELETED: position encoding
        # self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        pe_slice = max(16, min(64, filts[-1][-1] // 4)) 
        self.pe_proj = nn.Linear(pe_slice + 1, filts[-1][-1])  # [sin/cos_slice, t/T] -> C
        nn.init.zeros_(self.pe_proj.weight)  
        nn.init.zeros_(self.pe_proj.bias)
        self.pe_gate_S = nn.Parameter(torch.tensor(0.10)) 
        self.pe_gate_T = nn.Parameter(torch.tensor(0.10))  
        self.pe_norm_S = nn.LayerNorm(filts[-1][-1])       
        self.pe_norm_T = nn.LayerNorm(filts[-1][-1])
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # HGNN module - replaces the original graph module with integrated FuzzyCMeans
        self.HGNN_layer_S = HypergraphConvLayer(filts[-1][-1],
                                               gat_dims[0],
                                               n_clusters=16,
                                               temperature=temperatures[0])
        self.HGNN_layer_T = HypergraphConvLayer(filts[-1][-1],
                                               gat_dims[0],
                                               n_clusters=16,
                                               temperature=temperatures[1])
        # Heterogeneous HGNN layer - replaces the original HS-GAL layer with integrated FuzzyCMeans
        self.HtrgHGNN_layer_ST11 = HeterogeneousHGNNLayer(
            gat_dims[0], gat_dims[1], n_clusters=20, temperature=temperatures[2])
        self.HtrgHGNN_layer_ST12 = HeterogeneousHGNNLayer(
            gat_dims[1], gat_dims[1], n_clusters=16, temperature=temperatures[2])
        self.HtrgHGNN_layer_ST21 = HeterogeneousHGNNLayer(
            gat_dims[0], gat_dims[1], n_clusters=20, temperature=temperatures[2])
        self.HtrgHGNN_layer_ST22 = HeterogeneousHGNNLayer(
            gat_dims[1], gat_dims[1], n_clusters=16, temperature=temperatures[2])
        
        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

        # Create a shared manager (prior pos=0.1 comes from the 1:9 ratio)
        self.proto_manager = ProtoMemoryManager(
            prior_pos=0.1,
            proto_momentum=0.1,
            global_momentum=0.1,
            gate_tau=0.25,
            jitter_std=0.01,
            fuse_gamma=0.3,
            warm_on_eval=True,
            device=self.device,
        )

        # Bind each FCM-enabled layer to the manager with a unique key
        def bind(layer, key):
            if hasattr(layer, "fcm_module"):  # Only bind layers that include FCM
                layer.proto_manager = self.proto_manager
                layer.proto_key = key
        bind(self.HGNN_layer_S,   "HGNN_layer_S")
        bind(self.HGNN_layer_T,   "HGNN_layer_T")
        bind(self.HtrgHGNN_layer_ST11, "HtrgHGNN_layer_ST11")
        bind(self.HtrgHGNN_layer_ST12, "HtrgHGNN_layer_ST12")
        bind(self.HtrgHGNN_layer_ST21, "HtrgHGNN_layer_ST21")
        bind(self.HtrgHGNN_layer_ST22, "HtrgHGNN_layer_ST22")

        # Override the cleanup helper
        def clear_external_inits():
            for layer in [self.HGNN_layer_S, self.HGNN_layer_T,
                         self.HtrgHGNN_layer_ST11, self.HtrgHGNN_layer_ST12,
                         self.HtrgHGNN_layer_ST21, self.HtrgHGNN_layer_ST22]:
                if hasattr(layer, "fcm_module"):
                    layer.fcm_module.external_init_centroids = None
                if hasattr(layer, "hgnn_conv") and hasattr(layer.hgnn_conv, "fcm_module"):
                    layer.hgnn_conv.fcm_module.external_init_centroids = None
        self.proto_manager._clear_all_external_inits = clear_external_inits

    def forward(self, x, lengths=None):
        """
        Forward pass.
        Args:
            x: input waveform (batch_size, time) or (batch_size, time, 1)
            lengths: original waveform lengths (batch_size,) for variable-length support
        """
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        # x_ssl_feat = self.module.ssl_model.extract_feat(x.squeeze(-1))
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        
        # If lengths are provided, build temporal masks (supports variable-length inputs)
        time_mask_T = None  # Mask along the temporal dimension
        time_mask_S = None  # Mask along the spectral dimension
        if lengths is not None:
            # SSL front-end changes the time dimension, so estimate the scaling factor
            time_reduction = x.size(1) / lengths.float().mean()
            scaled_lengths = (lengths * time_reduction).long()
            # mask: (B, T) True indicates a valid position
            max_time = x.size(1)
            time_mask_raw = torch.arange(max_time, device=x.device)[None, :] < scaled_lengths[:, None]
        else:
            time_mask_raw = None
        
        # post-processing on front-end features
        x = x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1) # add channel 
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)
        
        # Generate masks (based on the current feature map shape)
        if time_mask_raw is not None:
            B_cur, C_cur, H_cur, W_cur = x.shape
            # Spectral-dimension mask (maps to H)
            time_mask_S = torch.ones(B_cur, H_cur, device=x.device, dtype=torch.bool)
            # Temporal-dimension mask (maps to W)
            time_mask_T = torch.ones(B_cur, W_cur, device=x.device, dtype=torch.bool)
            # Simplified handling: mark everything as valid if padding existed
        else:
            time_mask_S = None
            time_mask_T = None

        #------------SA for spectral feature-------------#
        w1 = F.softmax(w,dim=-1)
        # m = torch.sum(x * w1, dim=-1)
        # e_S = m.transpose(1, 2) + self.pos_S 
        m = torch.sum(x * w1, dim=-1)           # (B, C, N_S)
        e_S = m.transpose(1, 2)                 # (B, N_S, C)
        B, N_S, C = e_S.shape
        device, dtype = e_S.device, e_S.dtype

        # Dynamically generate the restrained positional encoding
        sin_pe_S = sinusoidal_pos_encoding(N_S, max(16, min(64, C // 4)), device, dtype)
        tcoord_S = normalized_time_coords(N_S, device, dtype)
        pe_in_S  = torch.cat([sin_pe_S.expand(B,-1,-1), tcoord_S.expand(B,-1,-1)], dim=-1)
        pe_S     = self.pe_proj(pe_in_S)
        # Inject via small gates and layer norm (additive to avoid widening the channel)
        e_S = self.pe_norm_S(e_S + self.pe_gate_S * pe_S)

        # HGNN module layer - replaces the original graph module
        hgnn_S = self.HGNN_layer_S(e_S)
        out_S = self.pool_S(hgnn_S)

        #------------SA for temporal feature-------------#
        w2 = F.softmax(w,dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
     
        e_T = m1.transpose(1, 2)
        B, N_T, C = e_T.shape
        sin_pe_T = sinusoidal_pos_encoding(N_T, max(16, min(64, C // 4)), device, dtype)
        tcoord_T = normalized_time_coords(N_T, device, dtype)
        pe_in_T  = torch.cat([sin_pe_T.expand(B,-1,-1), tcoord_T.expand(B,-1,-1)], dim=-1)
        pe_T     = self.pe_proj(pe_in_T)
        e_T = self.pe_norm_T(e_T + self.pe_gate_T * pe_T)

        # HGNN module layer - replaces the original graph module
        hgnn_T = self.HGNN_layer_T(e_T)
        out_T = self.pool_T(hgnn_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1 - heterogeneous HGNN replaces the original heterogeneous GAT
        out_T1, out_S1, master1 = self.HtrgHGNN_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgHGNN_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2 - heterogeneous HGNN replaces the original heterogeneous GAT
        out_T2, out_S2, master2 = self.HtrgHGNN_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgHGNN_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation with masking support
        if time_mask_T is not None and time_mask_S is not None:
            # Apply masks along the time dimension
            # Expand masks to the feature dimension
            mask_T_expanded = time_mask_T.unsqueeze(-1).float()
            mask_S_expanded = time_mask_S.unsqueeze(-1).float()

            # Masked max: set padded positions to a very small value
            T_max = torch.max(out_T + (1 - mask_T_expanded) * -1e9, dim=1)[0]
            S_max = torch.max(out_S + (1 - mask_S_expanded) * -1e9, dim=1)[0]

            # Masked average: only aggregate valid positions
            T_sum = (out_T * mask_T_expanded).sum(dim=1)
            T_count = mask_T_expanded.sum(dim=1).clamp(min=1.0)
            T_avg = T_sum / T_count

            S_sum = (out_S * mask_S_expanded).sum(dim=1)
            S_count = mask_S_expanded.sum(dim=1).clamp(min=1.0)
            S_avg = S_sum / S_count
        else:
            # Original logic (no masks)
            T_max, _ = torch.max(torch.abs(out_T), dim=1)
            T_avg = torch.mean(out_T, dim=1)
            S_max, _ = torch.max(torch.abs(out_S), dim=1)
            S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        return output