import torch
import torch.nn.functional as F
import torch.distributed as dist


class ProtoMemoryManager:
    """
    Prototype memory manager for multiple clustering blocks.

    Each bank entry is a dict:
        bank = {
            "pos":    Tensor[K, C],
            "neg":    Tensor[K, C],
            "global": Tensor[K, C],
            "inited": Tensor[()] bool-like,
        }

    The manager is agnostic to the internal implementation of FCM modules.
    Warm-start is provided by setting `fcm_module.external_init_centroids` to (K, C).
    """

    def __init__(
        self,
        prior_pos: float = 0.1,
        proto_momentum: float = 0.1,
        global_momentum: float = 0.1,
        gate_tau: float = 0.0,
        jitter_std: float = 0.001,
        fuse_gamma: float = 0.3,
        warm_on_eval: bool = True,
        device: str = "cpu",
        use_slot_alignment: bool = True,
    ) -> None:
        self.prior_pos = float(prior_pos)
        self.mu = float(proto_momentum)          # EMA momentum for pos/neg
        self.beta = float(global_momentum)       # EMA momentum for global
        self.gate_tau = float(gate_tau)          # warm-start gating threshold
        self.jitter_std = float(jitter_std)      # warm-start perturbation scale
        self.fuse_gamma = float(fuse_gamma)      # fusion factor for global update

        self.warm_on_eval = bool(warm_on_eval)
        # Training-time warm-start is controlled externally if needed.
        self.warm_on_train = False

        self.use_slot_alignment = bool(use_slot_alignment)
        self.device = device

        self.banks = {}
        self.current_labels = None

    # ---------------------------------------------------------------------
    # Batch label handling
    # ---------------------------------------------------------------------
    def set_batch_labels(self, y: torch.Tensor, mask= None) -> None:
        """
        Register labels for the current batch.

        Parameters
        ----------
        y:
            Utterance-level labels of shape (B,) or (B,1),
            or node-level labels of shape (B,N) with values in {0,1}.
            If node-level labels are provided, the first node label is used as
            an utterance label.
        mask:
            Optional node mask of shape (B,N). Reserved for future extensions.
        """
        self.current_labels = {"y": y, "mask": mask}
        self._clear_all_external_inits()

    def _clear_all_external_inits(self) -> None:
        """
        Clear external centroid initializations in all FCM modules.

        This method is intended to be overridden by the model wrapper. The manager
        itself does not own FCM modules.
        """
        pass

    def clear_labels(self) -> None:
        """Clear cached labels and external initializations."""
        self.current_labels = None
        self._clear_all_external_inits()

    # ---------------------------------------------------------------------
    # Bank utilities
    # ---------------------------------------------------------------------
    def _ensure_bank(
        self, key: str, K: int, C: int, dtype: torch.dtype, device: torch.device
    ):
        """Create a bank for `key` if absent; otherwise return existing bank."""
        if key not in self.banks:
            self.banks[key] = {
                "pos": torch.zeros(K, C, device=device, dtype=dtype),
                "neg": torch.zeros(K, C, device=device, dtype=dtype),
                "global": torch.zeros(K, C, device=device, dtype=dtype),
                "inited": torch.tensor(False, device=device),
            }
        return self.banks[key]

    @staticmethod
    def _greedy_cosine_match(src: torch.Tensor, ref: torch.Tensor):
        """
        Greedy cosine matching between `src` and `ref` prototypes.

        Parameters
        ----------
        src, ref:
            Tensors of shape (K, C).

        Returns
        -------
        src_reordered:
            `src` reordered by the computed permutation.
        perm:
            Permutation indices (shape: (K,)).
        """
        K = src.size(0)
        sim = F.normalize(src, dim=-1) @ F.normalize(ref, dim=-1).T  # (K,K)

        used_r, used_c, perm = set(), set(), []
        for _ in range(K):
            i, j = divmod(sim.argmax().item(), K)
            if i in used_r or j in used_c:
                sim[i, j] = -1e9
                continue
            perm.append(i)
            used_r.add(i)
            used_c.add(j)
            sim[i, :] = -1e9
            sim[:, j] = -1e9

        perm_t = torch.tensor(perm, device=src.device, dtype=torch.long)
        return src.index_select(0, perm_t), perm_t

    @staticmethod
    def _class_conditional_centroids(
        X: torch.Tensor,
        U: torch.Tensor,
        y_utt: torch.Tensor,
        m: float = 2.0,
        eps: float = 1e-6,
    ):
        """
        Compute class-conditional centroids using utterance-level labels.

        Shapes
        ------
        X:     (B, N, C)
        U:     (B, N, K)
        y_utt: (B,) with values in {0,1}

        Returns
        -------
        C_pos: (B, K, C)
        W_pos: (B, K)   (sum of weights over N)
        C_neg: (B, K, C)
        W_neg: (B, K)
        """
        B, N, C = X.shape
        _ = C
        K = U.size(-1)

        y_utt = y_utt.float().view(B, 1)  # (B,1)
        y_pos = y_utt.expand(B, N).unsqueeze(-1)          # (B,N,1)
        y_neg = (1.0 - y_utt).expand(B, N).unsqueeze(-1)  # (B,N,1)

        W_pos = (y_pos * U).pow(m)  # (B,N,K)
        W_neg = (y_neg * U).pow(m)  # (B,N,K)

        num_pos = torch.bmm(W_pos.transpose(1, 2), X)  # (B,K,C)
        den_pos = W_pos.sum(dim=1, keepdim=True).transpose(1, 2) + eps  # (B,K,1)

        num_neg = torch.bmm(W_neg.transpose(1, 2), X)  # (B,K,C)
        den_neg = W_neg.sum(dim=1, keepdim=True).transpose(1, 2) + eps  # (B,K,1)

        C_pos = num_pos / den_pos
        C_neg = num_neg / den_neg
        return C_pos, W_pos.sum(dim=1), C_neg, W_neg.sum(dim=1)

    def _jitter(self, T: torch.Tensor) -> torch.Tensor:
        """Apply small Gaussian perturbation to prototypes."""
        if self.jitter_std <= 0:
            return T
        scale = self.jitter_std * (T.norm(dim=-1, keepdim=True) + 1e-6)
        return T + torch.randn_like(T) * scale

    # ---------------------------------------------------------------------
    # Warm-start
    # ---------------------------------------------------------------------
    def maybe_warm_start(self, key: str, fcm_module, X: torch.Tensor, K: int) -> None:
        """
        Optionally initialize FCM centroids from the global prototype bank.

        The warm-start is gated by cosine similarity between:
            - current batch mean feature
            - mean of the stored global bank

        If gating passes, `fcm_module.external_init_centroids` is set to a tensor
        of shape (K, C).
        """
        bank = self._ensure_bank(key, K, X.size(-1), X.dtype, X.device)
        if not bool(bank["inited"]): return
        if not self.warm_on_eval:    return

        with torch.no_grad():
            X_mean = X.detach().mean(dim=(0,1))  # (C,)
            bank_mean = bank["global"].mean(dim=0)  # (C,)

            X_norm = F.normalize(X_mean.unsqueeze(0), dim=1)
            bank_norm = F.normalize(bank_mean.unsqueeze(0), dim=1)
            sim_gate = torch.mm(X_norm, bank_norm.t()).item()
        
        if sim_gate < self.gate_tau:
            return
        
        p_pos = None
        if self.current_labels is not None and "y" in self.current_labels:
            y = self.current_labels["y"]
            if isinstance(y, torch.Tensor):
                if y.dim() == 2 and y.size(1) == 1: y = y.squeeze(1)
                p_pos = y.float().mean().item()

        ext = self._assemble_init(bank, K, p_pos=p_pos, beta=0.8)
        ext = self._jitter(ext).detach().clone()  # (K,C)
        fcm_module.external_init_centroids = ext

    def _assemble_init(self, bank, K: int, p_pos = None, beta: float = 0.8) -> torch.Tensor:
        """
        Assemble an initialization set of centroids.

        If p_pos is None:
            sample from global bank
        Else:
            sample from pos/neg banks proportionally, then fill with global bank
        """
        if p_pos is None:
            g = bank["global"]
            if g.size(0) <= K:
                return g
            perm = torch.randperm(g.size(0), device=g.device)
            return g[perm[:K]]

        parts = []
        k_pos = int(K * beta * float(p_pos))
        k_neg = int(K * beta * (1.0 - float(p_pos)))

        if k_pos > 0 and bank.get("pos", None) is not None:
            pos_bank = bank["pos"]
            n_pos = min(k_pos, pos_bank.size(0))
            perm_pos = torch.randperm(pos_bank.size(0), device=pos_bank.device)
            parts.append(pos_bank[perm_pos[:n_pos]])

        if k_neg > 0 and bank.get("neg", None) is not None:
            neg_bank = bank["neg"]
            n_neg = min(k_neg, neg_bank.size(0))
            perm_neg = torch.randperm(neg_bank.size(0), device=neg_bank.device)
            parts.append(neg_bank[perm_neg[:n_neg]])

        parts.append(bank["global"])
        out = torch.cat(parts, dim=0)

        if out.size(0) >= K:
            perm_all = torch.randperm(out.size(0), device=out.device)
            return out[perm_all[:K]]

        # Fallback: sample with replacement if insufficient.
        idx = torch.randint(0, out.size(0), (K,), device=out.device)
        return out.index_select(0, idx)

    # ---------------------------------------------------------------------
    # DDP synchronization
    # ---------------------------------------------------------------------
    def sync_banks(self, reduce_op: str = "mean") -> None:
        """
        Synchronize banks across distributed processes.

        - pos/neg/global are all-reduced by SUM.
        - If reduce_op == "mean", divide by world size.
        - inited is set to True if any rank has initialized the bank.
        """
        if not dist.is_available() or not dist.is_initialized():
            return

        world_size = dist.get_world_size()

        # Keys are sorted to ensure consistent collective call order across ranks.
        for key in sorted(self.banks.keys()):
            bank = self.banks[key]
            for name in ("pos", "neg", "global"):
                t = bank.get(name, None)
                if t is None or t.numel() == 0:
                    continue
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                if reduce_op == "mean" and world_size > 0:
                    t.div_(float(world_size))

            flag = bank["inited"].to(dtype=torch.int64).view(1)
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            bank["inited"].fill_(bool(flag.item() > 0))

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def save_banks(self, save_path: str) -> None:
        """Serialize all banks to a file."""
        banks_to_save = {}
        for key, bank in self.banks.items():
            banks_to_save[key] = {
                "pos": bank.get("pos", None).detach().cpu() if isinstance(bank.get("pos", None), torch.Tensor) else None,
                "neg": bank.get("neg", None).detach().cpu() if isinstance(bank.get("neg", None), torch.Tensor) else None,
                "global": bank.get("global", None).detach().cpu() if isinstance(bank.get("global", None), torch.Tensor) else None,
                "inited": bank.get("inited", torch.tensor(False)).detach().cpu()
                if isinstance(bank.get("inited", None), torch.Tensor)
                else torch.tensor(bool(bank.get("inited", False))),
            }
        torch.save(banks_to_save, save_path)
        print(f"[info] Saved prototype banks to {save_path} (num_keys={len(banks_to_save)})")

    def load_banks(self, banks_path: str) -> None:
        """Load serialized banks from a file."""
        saved = torch.load(banks_path, map_location=self.device)

        loaded_cnt = 0
        for key, bank_data in saved.items():
            loaded_bank = {}
            for tkey in ['pos', 'neg', 'global']:
                t = bank_data[tkey]
                loaded_bank[tkey] = t.to(self.device) if isinstance(t, torch.Tensor) else None

            # Mark as initialized if present in the checkpoint.
            loaded_bank['inited'] = True

            self.banks[key] = loaded_bank
            loaded_cnt += 1
            print(f"[info] 加载层 {key} 完成：pos{loaded_bank['pos'].shape}, neg{loaded_bank['neg'].shape}, global{loaded_bank['global'].shape}")

        print(f"[info] 原型库加载完成，共 {loaded_cnt} 层")

    # ---------------------------------------------------------------------
    # Updates (raw alignment)
    # ---------------------------------------------------------------------
    def update_raw(self, key: str, X: torch.Tensor, C_glob: torch.Tensor, U: torch.Tensor, m: float) -> None:
        """
        Update banks using a greedy cosine alignment against the stored global bank.

        Parameters
        ----------
        X:
            Node features, shape (B, N, C).
        C_glob:
            Per-sample global centroids, shape (B, K, C).
        U:
            Soft memberships, shape (B, N, K).
        m:
            FCM fuzzifier.
        """
        B, K, C = C_glob.shape
        bank = self._ensure_bank(key, K, C, X.dtype, X.device)

        X_detached = X.detach()
        C_glob_detached = C_glob.detach()
        U_detached = U.detach()

        Cg_mean = C_glob_detached.mean(dim=0)  # (K,C)

        # First-time initialization
        if not bool(bank["inited"]):
            bank["pos"].copy_(Cg_mean)
            bank["neg"].copy_(Cg_mean)
            bank["global"].copy_(Cg_mean)
            bank["inited"].fill_(True)

        labels = self.current_labels
        if labels is None:
            bank["global"] = (1 - self.beta) * bank["global"] + self.beta * Cg_mean
            return

        y_utt = labels["y"].to(X_detached.device)
        B, N, C = X_detached.shape

        # Normalize y_utt to (B,)
        if y_utt.dim() == 2 and y_utt.size(1) == 1:
            y_utt = y_utt.squeeze(1)
        elif y_utt.dim() == 2 and y_utt.size(1) == N:
            y_utt = y_utt[:, 0]

        assert y_utt.dim() == 1 and y_utt.size(0) == B, (
            f"Expected utterance labels of shape (B,) with B={B}, got {tuple(y_utt.shape)}"
        )

        C_pos, Wp, C_neg, Wn = self._class_conditional_centroids(X_detached, U_detached, y_utt, m=m)
        has_pos = bool(Wp.sum() > 0)
        has_neg = bool(Wn.sum() > 0)

        def _weighted_mean(C_bkc: torch.Tensor, W_bk: torch.Tensor) -> torch.Tensor:
            Wsum = W_bk.sum(dim=0).clamp_min(1e-6)           # (K,)
            num = (C_bkc * W_bk.unsqueeze(-1)).sum(dim=0)   # (K,C)
            return num / Wsum.unsqueeze(-1)

        Cp_mean = _weighted_mean(C_pos, Wp) if has_pos else None
        Cn_mean = _weighted_mean(C_neg, Wn) if has_neg else None

        # Align batch prototypes to the stored global slots.
        Cg_aligned, perm = self._greedy_cosine_match(Cg_mean, bank["global"])
        if has_pos:
            Cp_aligned = Cp_mean.index_select(0, perm)
        if has_neg:
            Cn_aligned = Cn_mean.index_select(0, perm)

        mu, beta, gamma = self.mu, self.beta, self.fuse_gamma

        if has_pos:
            bank["pos"] = (1 - mu) * bank["pos"] + mu * Cp_aligned
        if has_neg:
            bank["neg"] = (1 - mu) * bank["neg"] + mu * Cn_aligned

        neutral_bal = 0.5 * (bank["pos"] + bank["neg"])
        bank["global"] = (1 - beta) * bank["global"] + beta * (gamma * Cg_aligned.detach() + (1 - gamma) * neutral_bal)

    # ---------------------------------------------------------------------
    # Updates (slot assignment)
    # ---------------------------------------------------------------------
    def update_slot(self, key: str, X: torch.Tensor, C_glob: torch.Tensor, U: torch.Tensor, m: float = 2.0) -> None:
        """
        Update banks using soft slot assignment to the stored global prototypes.

        Each local centroid (from each sample and each cluster) is softly assigned
        to K global slots based on cosine similarity. Aggregation is performed
        in slot space, followed by EMA updates.

        Parameters
        ----------
        X:
            Node features, shape (B, N, C).
        C_glob:
            Per-sample global centroids, shape (B, K, C).
        U:
            Soft memberships, shape (B, N, K).
        m:
            FCM fuzzifier.
        """
        B, K, C = C_glob.shape
        bank = self._ensure_bank(key, K, C, X.dtype, X.device)

        X_detached = X.detach()
        C_glob_detached = C_glob.detach()
        U_detached = U.detach()

        Cg_mean = C_glob_detached.mean(dim=0)  # (K,C)

        # First-time initialization
        if not bool(bank["inited"]):
            bank["pos"].copy_(Cg_mean)
            bank["neg"].copy_(Cg_mean)
            bank["global"].copy_(Cg_mean)
            bank["inited"].fill_(True)

        M = B * K
        C_flat = C_glob_detached.view(M, C)      # (M,C)
        global_proto = bank["global"].detach()   # (K,C)

        tau = 0.5
        sim = F.normalize(C_flat, dim=-1) @ F.normalize(global_proto, dim=-1).T  # (M,K)
        A = F.softmax(sim / tau, dim=-1)  # (M,K)

        def aggregate_slots_soft(
            values_flat: torch.Tensor,
            weights_flat = None,
            base = None,
        ) -> torch.Tensor:
            """
            Aggregate flattened values into K slots using soft assignments A.

            values_flat:  (M,C)
            weights_flat: (M,) optional
            base:         (K,C) fallback for empty slots
            """
            if base is None:
                base = torch.zeros(K, C, device=values_flat.device, dtype=values_flat.dtype)
            if weights_flat is None:
                weights_flat = torch.ones(M, device=values_flat.device, dtype=values_flat.dtype)

            A_weighted = A * weights_flat.view(-1, 1)      # (M,K)
            num = A_weighted.T @ values_flat               # (K,C)
            den = A_weighted.sum(dim=0, keepdim=True).T    # (K,1)

            out = base.clone()
            valid = den.squeeze(-1) > 0
            out[valid] = num[valid] / den[valid]
            return out

        # 1) Slot-aligned global prototypes for the current batch
        Cg_slots = aggregate_slots_soft(C_flat, weights_flat=None, base=bank["global"])
        lambda_mix = 0.5
        Cg_batch = lambda_mix * Cg_slots + (1 - lambda_mix) * Cg_mean

        labels = self.current_labels
        mu, beta, gamma = self.mu, self.beta, self.fuse_gamma

        # If labels are unavailable, update only the global bank.
        if labels is None or ("y" not in labels) or (labels["y"] is None):
            bank["global"] = (1 - beta) * bank["global"] + beta * Cg_batch
            return

        y_utt = labels["y"].to(X_detached.device)
        N = X_detached.size(1)

        # Normalize y_utt to (B,)
        if y_utt.dim() == 2 and y_utt.size(1) == 1:
            y_utt = y_utt.squeeze(1)
        elif y_utt.dim() == 2 and y_utt.size(1) == N:
            y_utt = y_utt[:, 0]

        assert y_utt.dim() == 1 and y_utt.size(0) == B, (
            f"Expected utterance labels of shape (B,) with B={B}, got {tuple(y_utt.shape)}"
        )

        C_pos, Wp, C_neg, Wn = self._class_conditional_centroids(X_detached, U_detached, y_utt, m=m)
        C_pos_flat = C_pos.view(M, C)
        C_neg_flat = C_neg.view(M, C)
        Wp_flat = Wp.view(M)
        Wn_flat = Wn.view(M)

        has_pos = bool(Wp_flat.sum() > 0)
        has_neg = bool(Wn_flat.sum() > 0)

        if has_pos:
            Cp_batch = aggregate_slots_soft(C_pos_flat, weights_flat=Wp_flat, base=bank["pos"])
            bank["pos"] = (1 - mu) * bank["pos"] + mu * Cp_batch

        if has_neg:
            Cn_batch = aggregate_slots_soft(C_neg_flat, weights_flat=Wn_flat, base=bank["neg"])
            bank["neg"] = (1 - mu) * bank["neg"] + mu * Cn_batch

        neutral_bal = 0.5 * (bank["pos"] + bank["neg"])
        bank["global"] = (1 - beta) * bank["global"] + beta * (gamma * Cg_batch.detach() + (1 - gamma) * neutral_bal)
