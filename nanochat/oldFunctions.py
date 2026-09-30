#In gpt, ortho optimizer setup original

def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, 
                        orthog_within_head=False, orthog_across_heads=False, concat_qk=False, stiefel=False):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        #stiefel params
        stiefel_params=[]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        #default, if orthogonality with all heads concatenated
        if not orthog_within_head and not orthog_across_heads and not stiefel:
            matrix_params = list(self.transformer.h.parameters())
        elif orthog_across_heads and not orthog_within_head:
            print("orthog_across_heads")
            matrix_params=[]
            ortho_params=[]
            o=0
            for h in self.transformer['h']:
                for n, p in h.named_parameters():
                    if "c_q" in n or "c_k" in n:
                        print(n, p.shape)
                        ortho_params.append(p)
                        o+=1
                    else:
                        matrix_params.append(p)
                #ortho_params.append(ortho)

            assert len(list(self.parameters())) == o + len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        elif orthog_within_head:
            if not concat_qk:
                print("orthog_within_head")
                matrix_params=[]
                ortho_params=[]
                q=[]
                k=[]
                v=[]
                for h in self.transformer['h']:
                    for n, p in h.named_parameters():
                        if "c_q" not in n and "c_k" not in n and "c_v" not in n:
                            matrix_params.append(p)
                        elif "c_q" in n:
                            q.append(p)
                        elif "c_k" in n:
                            k.append(p)
                        elif "c_v" in n:
                            v.append(p)
                ortho_params+=q
                ortho_params+=k
                ortho_params+=v

                assert len(list(self.parameters())) == len(q) + len(k) + len(v) + len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
            else:
                print("orthog_within_head, qk")
                matrix_params=[]
                ortho_params=[]
                qk=[]
                q=[]
                k=[]
                v=[]
                for h in self.transformer['h']:
                    for n, p in h.named_parameters():
                        if "c_q" not in n and "c_k" not in n and "c_v" not in n:
                            matrix_params.append(p)
                        elif "c_q" in n:
                            q.append(p)
                        elif "c_k" in n:
                            k.append(p)
                        elif "c_v" in n:
                            v.append(p)
                for i in range(len(q)):
                    qk+=[q[i], k[i]]
                ortho_params+=v

                assert len(list(self.parameters())) == len(qk) + len(ortho_params) + len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        elif stiefel:
            print("stiefel")
            matrix_params=[]
            q=[]
            k=[]
            for h in self.transformer['h']:
                for n, p in h.named_parameters():
                    if "c_q" not in n and "c_k" not in n:
                        matrix_params.append(p)
                    elif "c_q" in n:
                        q.append(p)
                    elif "c_k" in n:
                        k.append(p)
            stiefel_params+=q+k

            assert len(list(self.parameters())) == len(stiefel_params) + len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        if orthog_across_heads:
            param_groups.append(dict(kind='muon-ortho-across', params=ortho_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))

        if orthog_within_head and not concat_qk:
            param_groups.append(dict(kind='muon-ortho-within', params=ortho_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay, h=self.config.n_head, 
                d=self.config.n_embd // self.config.n_head, qk_together=False))
        elif orthog_within_head and concat_qk:
            param_groups.append(dict(kind='muon-ortho-within-qk', params=qk, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay, h=self.config.n_head, 
                d=self.config.n_embd // self.config.n_head, qk_together=True))
            param_groups.append(dict(kind='muon-ortho-within-v', params=ortho_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay, h=self.config.n_head, 
                d=self.config.n_embd // self.config.n_head, qk_together=True))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer, stiefel_params