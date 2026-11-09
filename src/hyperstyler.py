import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling_t5_for_hyperstyler_rerankcache import T5ForConditionalGeneration_HyperStyler


class HyperStyler(nn.Module):
    def __init__(self, base_model, config = None, use_style=False):
        super().__init__()
        self.model = T5ForConditionalGeneration_HyperStyler.from_pretrained(base_model)

        self.use_style = use_style

        if self.use_style:

            self.base_config = {
                    "d_model" : self.model.config.d_model,  #1024
                    "num_encoder_layers" : self.model.config.num_layers,
                    "num_decoder_layers" : self.model.config.num_decoder_layers,
                }
        
            self.config = config
            self.use_hypernet = config['use_hypernet']
            self.use_stylonavigator = config['use_stylonavigator']
            self.ctrl_dim = config['ctrl_dim']

            print(config)
            
            if self.use_stylonavigator:
                sn_config = config['stylo_navigator_config']
                self.style_navigator = StyleNavigator(self.base_config['d_model'],  config['ctrl_dim'], 
                                                      dropout=sn_config['dropout'] , 
                                                      temperature= sn_config['temperature'])

            if self.use_hypernet:
                hn_config = config['hypernet_config']
                self.hypernet = HyperNetwork(model_config=self.base_config, 
                                             hypernet_config=hn_config,
                                             ctrl_dim = self.ctrl_dim)
            else:
                hn_config = config['hypernet_config']
                self.internal_controller = InternalControl(model_config=self.base_config, 
                                                        hypernet_config=hn_config,
                                                        ctrl_dim = self.ctrl_dim)

    def freeze_stylonavigator(self):
        for name, param in self.style_navigator.named_parameters():
            param.requires_grad = False

    def base_model_freeze(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def freeze_encoder_only(self):
        for name, param in self.model.named_parameters():
            # encoder freeze
            if name.startswith("encoder."):
                param.requires_grad = False
            # shared embedding freeze
            elif name.startswith("shared."):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None, style_embs=None, style_masks=None, idx_labels=None,target_style_embs=None,):
        if self.use_style:
            stylonavigator = None
            hypernet = None
            internal_controller = None
            if self.use_stylonavigator:
                stylonavigator = self.style_navigator
            if self.use_hypernet:
                hypernet = self.hypernet
            else:
                internal_controller = self.internal_controller
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, 
                labels=labels, 
                style_embs = style_embs, style_masks = style_masks,
                style_navigator = stylonavigator,
                hypernetwork = hypernet,
                idx_labels=idx_labels,
                internal_controller= internal_controller,
                target_style_embs=target_style_embs,
            )
        else:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

    def generate(self, input_ids, attention_mask, style_embs=None, style_masks=None, **kwargs):
        if self.use_style:
            stylonavigator = None
            hypernet = None
            internal_controller = None
            if self.use_stylonavigator:
                stylonavigator = self.style_navigator
            if self.use_hypernet:
                hypernet = self.hypernet
            else:
                internal_controller = self.internal_controller

            return self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask , 
                style_embs = style_embs, style_masks = style_masks,
                style_navigator = stylonavigator,
                hypernetwork = hypernet,
                internal_controller= internal_controller, **kwargs
            )
        else:
            return self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )


class StyleNavigator(nn.Module):
    """
    Aggregates multiple author style embeddings (few-shot samples)
    into a single style summary embedding.
    Cosine-space normalization + learnable scaling for stability.
    """
    def __init__(self, d_model, style_dim, dropout=0.1 , temperature= 1.0):
        super().__init__()
        self.d_model = d_model
        self.style_dim = style_dim

        # --- Text Multi-Head Self-Attention ---
        self.cross_attn = MultiHeadAttention(d_model, style_dim, style_dim, style_dim, 16)
        self.style_attn = MultiHeadAttention(style_dim, style_dim, style_dim, style_dim, 16)

        self.style_layer_norm = nn.LayerNorm(style_dim)
        self.text_layer_norm = nn.LayerNorm(d_model)
        self.q_layer_norm = nn.LayerNorm(style_dim)
        self.k_layer_norm = nn.LayerNorm(style_dim)

        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, text_embs, style_embs, text_mask=None,  style_mask=None, idx_labels= None):

        #B, N, D = text_embs.shape
        norm_style_embs = self.style_layer_norm(style_embs)
        norm_text_embs = self.text_layer_norm(text_embs)
        query = self.cross_attn(norm_text_embs, norm_style_embs, norm_style_embs, text_mask, style_mask)
        query = self.mask_mean_pooling(query, text_mask).unsqueeze(1)

        key = self.style_attn(norm_style_embs, q_mask= style_mask, k_mask=style_mask)

        query = self.q_layer_norm(query)
        key = self.k_layer_norm(key)

        attn_logits = (query @ key.transpose(-2, -1)) / (key.size(-1) ** 0.5)

        if style_mask is not None:
            style_mask = style_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(style_mask == 0, -1e9)
        
        accuracy = None
        loss = None
        if idx_labels is not None:
            attn_probs = F.softmax(attn_logits, dim=-1)
            log_probs = F.log_softmax(attn_logits.squeeze(1), dim=-1)
            loss = nn.NLLLoss()(log_probs, idx_labels)
            pred_idx = torch.argmax(attn_probs.squeeze(1), dim=-1)
            accuracy = (pred_idx == idx_labels).float().mean()
            # for teacher forcing to later network
            selected_embs = style_embs.gather(1, idx_labels.view(-1, 1, 1).expand(-1, 1, style_embs.size(-1))).squeeze(1)
        else:
            attn_probs = F.softmax(attn_logits / self.temperature, dim=-1)
            #print("attn_probs : ", attn_probs.shape)
            #print("style_embs : ",style_embs.shape)
            selected_embs = (attn_probs @ style_embs).squeeze(1)
            #pred_idx = torch.argmax(attn_probs.squeeze(1), dim=-1)
            #selected_embs = style_embs.gather(1, pred_idx.view(-1, 1, 1).expand(-1, 1, style_embs.size(-1))).squeeze(1)

        return selected_embs, loss, accuracy, attn_probs

    def mask_mean_pooling(self, x, mask):
        if mask is not None:
            mask = mask.float().unsqueeze(-1)  # [B, Nq, 1]
            out = (x * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1e-6))
        else:
            out = x.mean(dim=1)
        return out

class BaseMLP(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, bias=False, out_layer_norm = False, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.final_out_dim = self.output_dim[0] * self.output_dim[1]

        self.out_layer_norm = out_layer_norm
        if out_layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_dim[1])

        self.fc = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # projection layer for weight / bias
        self.fc_weight = nn.Linear(intermediate_dim, self.final_out_dim)
        if bias:
            self.fc_bias = nn.Linear(intermediate_dim, self.final_out_dim)
            self._init_hyper_heads_bias()

        self._init_fc()
        self._init_hyper_heads() # bias_hyperinit

    def _init_fc(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_hyper_heads(self):
        #nn.init.constant_(self.fc_weight.weight, 0.0)
        #nn.init.xavier_uniform_(self.fc_weight.weight)
        nn.init.xavier_uniform_(self.fc_weight.weight, gain=0.5)
        nn.init.zeros_(self.fc_weight.bias)
        
    def _init_hyper_heads_bias(self):
        nn.init.xavier_uniform_(self.fc_bias.weight)
        nn.init.zeros_(self.fc_bias.bias)

    def forward(self, x):
        """
        x: [batch, num_layers, ctrl_dim]
        output: [batch, num_layers, *output_dim]
        """
        bsz, n_layer, d = x.shape

        h = self.fc(x.view(bsz * n_layer, d))                 # [batch*num_layers, hidden_dim]
        w = self.fc_weight(h)                                 # [batch*num_layers, final_out_dim]
        #b = self.fc_bias(h)                                 # [batch*num_layers, final_out_dim]
        w = w.view(bsz, n_layer, *self.output_dim)            # [batch, num_layers, *output_dim]
        #b = b.view(bsz, n_layer, *self.output_dim)            # [batch, num_layers, *output_dim]
        if self.out_layer_norm:
            w = self.layer_norm(w)
        return w #, b
    

class InternalControl(nn.Module):
    def __init__(self,  model_config, hypernet_config, ctrl_dim):
        super().__init__()
        self.layer_embeddings = HyperEmbeddingforInternalControl(ctrl_dim, model_config, hypernet_config, block_name="decoder")
        
        d_model = model_config['d_model']
        self.mlp = BaseMLP(ctrl_dim, d_model, (1, d_model),  dropout=hypernet_config['dropout'])


        self.model_config = model_config
        self.hypernet_config = hypernet_config

        self.q = nn.Linear(ctrl_dim, ctrl_dim, bias=False)
        if self.hypernet_config['search_config'] == "MHA":
            self.k = nn.Linear(ctrl_dim, ctrl_dim, bias=False)
            self.layers_layer_norm = nn.LayerNorm(ctrl_dim)
            self.style_layer_norm = nn.LayerNorm(ctrl_dim)

        self.v = nn.Linear(ctrl_dim, ctrl_dim, bias=False)
        self.o = nn.Linear(ctrl_dim, ctrl_dim, bias=False)

        self.out_layer_norm = nn.LayerNorm(ctrl_dim)

        self._init_weights()

    def _init_weights(self):
        # q, k, v, o projection
        target_layers = [self.q, self.v, self.o]
        if self.hypernet_config['search_config'] == "MHA":
            target_layers.append(self.k)
        for m in target_layers: # ,
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, author_embs):

        generated_prefix = {}
        layer_embs = self.layer_embeddings.embedding.weight 

        B = author_embs.size(0)
        L, D = layer_embs.shape
        H = 16
        d_h = D // H

        #Q = layer_embs.view(L, H, d_h).transpose(0, 1)
        ####
        norm_layer_embs = self.layers_layer_norm(layer_embs)#, layer_embs.shape[-1:])
        norm_style_embs = self.style_layer_norm(author_embs)#, style_embs.shape[-1:])
        Q = self.q(norm_layer_embs).view(L, H, d_h).transpose(0, 1)
        ####
        K = self.k(norm_style_embs).unsqueeze(1).view(B, 1, H, d_h).transpose(1, 2)
        V = self.v(norm_style_embs).unsqueeze(1).view(B, 1, H, d_h).transpose(1, 2)

        attn = torch.einsum("hld,bhnd->bhln", Q, K) #/ (d_h ** 0.5)
        out = torch.einsum("bhln,bhnd->bhld", attn, V)   
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o(out)
        out = self.out_layer_norm(out)
        # out : [batch, n_layers, dim]

        attn_layer_embs = layer_embs.unsqueeze(0) + out

        gen_query = self.mlp(attn_layer_embs).squeeze(-2) # B, N, D
        
        for i, layer_name in enumerate(self.layer_embeddings.q_name_list):
            generated_prefix[layer_name] = gen_query[:,i,:] # B, 1, D 

        return generated_prefix


class HyperNetwork(nn.Module):
    def __init__(self, model_config, hypernet_config, ctrl_dim):
        super().__init__()
        self.prefix_embeddings = None
        self.adapter_embeddings = None

        d_model = model_config['d_model']

        self.model_config = model_config
        self.hypernet_config = hypernet_config

        if 'selfattn' in hypernet_config['target_layers']:
            self.decoder_self_prefix_embeddings = HyperEmbeddingforPrefix(ctrl_dim, model_config, hypernet_config, block_name="decoder", layer_name="self")
            self.decoder_self_prefix_key = BaseMLP(ctrl_dim, d_model, (1, d_model), dropout=hypernet_config['dropout'] , out_layer_norm = True)
            self.decoder_self_prefix_value = BaseMLP(ctrl_dim, d_model, (1, d_model), dropout=hypernet_config['dropout'] , out_layer_norm = True)

        if 'crossattn' in hypernet_config['target_layers']:
            self.decoder_cross_prefix_embeddings = HyperEmbeddingforPrefix(ctrl_dim, model_config, hypernet_config, block_name="decoder", layer_name="cross")
            self.decoder_cross_prefix_key = BaseMLP(ctrl_dim, d_model, (1, d_model), dropout=hypernet_config['dropout'] , out_layer_norm = True)
            self.decoder_cross_prefix_value = BaseMLP(ctrl_dim, d_model, (1, d_model), dropout=hypernet_config['dropout'] , out_layer_norm = True)

        if 'ffn' in hypernet_config['target_layers']:
            self.decoder_adapter_embeddings = HyperEmbeddingforAdaptor(ctrl_dim, model_config, hypernet_config, block_name="decoder")

            if self.hypernet_config['parameter_efficiency_mode']:
                self.decoder_adaptor_Wu = BaseMLP(ctrl_dim, d_model, (1, d_model), dropout=hypernet_config['dropout'])
                self.decoder_adaptor_Wd = BaseMLP(ctrl_dim, d_model, (1, d_model),  dropout=hypernet_config['dropout'])

            else:
                self.decoder_adaptor_Wu = BaseMLP(ctrl_dim, d_model, (hypernet_config["rank"], d_model),  dropout=hypernet_config['dropout'])
                self.decoder_adaptor_Wd = BaseMLP(ctrl_dim, d_model, (d_model, hypernet_config["rank"]),  dropout=hypernet_config['dropout'])

        self.q = nn.Linear(ctrl_dim, ctrl_dim, bias=False)
        if self.hypernet_config['search_config'] == "MHA" or self.hypernet_config['search_config'] == "MHA_all_style_embeddings":
            self.k = nn.Linear(ctrl_dim, ctrl_dim, bias=False)
            self.layers_layer_norm = nn.LayerNorm(ctrl_dim)
            self.style_layer_norm = nn.LayerNorm(ctrl_dim)

        self.v = nn.Linear(ctrl_dim, ctrl_dim, bias=False)
        self.o = nn.Linear(ctrl_dim, ctrl_dim, bias=False)

        self.out_layer_norm = nn.LayerNorm(ctrl_dim)

        self._init_weights()
            
    def _init_weights(self):
        # q, k, v, o projection
        target_layers = [self.q, self.v, self.o]
        if self.hypernet_config['search_config'] == "MHA" or self.hypernet_config['search_config'] == "MHA_all_style_embeddings":
            target_layers.append(self.k)
        for m in target_layers: # ,
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_inputs_multiheadattn_for_ablation(self, embedding_module, style_embs, style_masks):
        layer_embs = embedding_module.embedding.weight  # [L_q, D]

        B, L_k, D = style_embs.shape
        L_q = layer_embs.size(0)

        H = 16
        d_h = D // H

        norm_layer_embs = self.layers_layer_norm(layer_embs)
        norm_style_embs = self.style_layer_norm(style_embs)

        # Q: [B, H, L_q, d_h]
        Q = self.q(norm_layer_embs) \
                .view(L_q, H, d_h).transpose(0, 1) \
                .unsqueeze(0).expand(B, -1, -1, -1)

        # K, V: [B, H, L_k, d_h]
        K = self.k(norm_style_embs) \
                .view(B, L_k, H, d_h).transpose(1, 2)

        V = self.v(norm_style_embs) \
                .view(B, L_k, H, d_h).transpose(1, 2)

        attn_logits = torch.einsum("bhld,bhnd->bhln", Q, K)
        attn_logits = attn_logits / (d_h ** 0.5)

        style_masks = style_masks.unsqueeze(1).unsqueeze(2)  # [B,1,1,L_k]
        attn_logits = attn_logits.masked_fill(style_masks == 0, float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)

        out = torch.einsum("bhln,bhnd->bhld", attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)

        out = self.o(out)
        out = self.out_layer_norm(out)

        return layer_embs.unsqueeze(0) + out

    def get_inputs_multihead(self, embedding_module, style_embs):
        layer_embs = embedding_module.embedding.weight 

        B = style_embs.size(0)
        L, D = layer_embs.shape
        H = 16
        d_h = D // H

        #Q = layer_embs.view(L, H, d_h).transpose(0, 1)
        ####
        norm_layer_embs = self.layers_layer_norm(layer_embs)#, layer_embs.shape[-1:])
        norm_style_embs = self.style_layer_norm(style_embs)#, style_embs.shape[-1:])
        Q = self.q(norm_layer_embs).view(L, H, d_h).transpose(0, 1)
        ####
        K = self.k(norm_style_embs).unsqueeze(1).view(B, 1, H, d_h).transpose(1, 2)
        V = self.v(norm_style_embs).unsqueeze(1).view(B, 1, H, d_h).transpose(1, 2)

        attn = torch.einsum("hld,bhnd->bhln", Q, K) #/ (d_h ** 0.5)
        out = torch.einsum("bhln,bhnd->bhld", attn, V)   
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o(out)
        out = self.out_layer_norm(out)
        # out : [batch, n_layers, dim]

        return layer_embs.unsqueeze(0) + out


    def get_inputs(self, embedding_module, style_embs, style_masks=None):
        if  self.hypernet_config['search_config'] == "MHA":
            return self.get_inputs_multihead(embedding_module, style_embs)
        elif self.hypernet_config['search_config'] == "MHA_all_style_embeddings":
            return self.get_inputs_multiheadattn_for_ablation(embedding_module, style_embs, style_masks)
        elif self.hypernet_config['search_config'] == "none":
            return embedding_module + style_embs
        else:
            # [L, D]
            layer_embs = embedding_module.embedding.weight  

            Q = self.q(layer_embs)          # [L, D]
            Q = F.normalize(Q, dim=-1)      # remaining direction

            # K, V: style direction 
            norm_style_embs = F.normalize(style_embs, dim=-1)
            K = norm_style_embs.unsqueeze(1)  # [B, 1, D]
            V = self.v(norm_style_embs).unsqueeze(1)  # [B, 1, D]

            # Attention: layer–style alignment
            # Q: [L, D], K: [B, 1, D] → attn: [B, L, 1]
            attn = torch.einsum("ld,bnd->bln", Q, K)

            # Modulation
            # attn: [B, L, 1], V: [B, 1, D] → out: [B, L, D]
            out = torch.einsum("bln,bnd->bld", attn, V)

            out = self.o(out)
            out = self.out_layer_norm(out)

            # Residual connection
            return layer_embs.unsqueeze(0) + out


    def forward(self, author_embs, style_masks=None): #style_embs, style_masks):
        # style_embs: [batch, num_style_embs, d]
        
        generated_parameters = {}
        for layer in self.hypernet_config['target_layers']:
            if layer == "selfattn":
                generated_parameters['is_selfattn'] = True
            if layer == "crossattn":
                generated_parameters['is_crossattn'] = True
            if layer == "ffn":
                generated_parameters['is_ffn'] = True

        #author_embs = self.author_encoder(style_embs, style_masks)

        if 'selfattn' in self.hypernet_config['target_layers']:
            dec_self_embs = self.get_inputs(self.decoder_self_prefix_embeddings, author_embs, style_masks)
            dec_self_prefix = self.generate_prefix_params(
                dec_self_embs,
                self.decoder_self_prefix_key, self.decoder_self_prefix_value,
                self.decoder_self_prefix_embeddings.key_name_list, self.decoder_self_prefix_embeddings.value_name_list,
                self.decoder_self_prefix_embeddings.key_idx_list, self.decoder_self_prefix_embeddings.value_idx_list,
            )
            generated_parameters.update(dec_self_prefix)


        if 'crossattn' in self.hypernet_config['target_layers']:
            dec_cross_embs = self.get_inputs(self.decoder_cross_prefix_embeddings, author_embs, style_masks)
            dec_cross_prefix = self.generate_prefix_params(
                dec_cross_embs,
                self.decoder_cross_prefix_key, self.decoder_cross_prefix_value,
                self.decoder_cross_prefix_embeddings.key_name_list, self.decoder_cross_prefix_embeddings.value_name_list,
                self.decoder_cross_prefix_embeddings.key_idx_list, self.decoder_cross_prefix_embeddings.value_idx_list,
            )
            generated_parameters.update(dec_cross_prefix)

        if 'ffn' in self.hypernet_config['target_layers']:
            dec_adapt_embs = self.get_inputs(self.decoder_adapter_embeddings, author_embs, style_masks)
            dec_adaptor_params = self.generate_adaptor_params(
                dec_adapt_embs,
                self.decoder_adaptor_Wu, self.decoder_adaptor_Wd,
                self.decoder_adapter_embeddings.Wu_name_list, self.decoder_adapter_embeddings.Wd_name_list,
                self.decoder_adapter_embeddings.Wu_idx_list, self.decoder_adapter_embeddings.Wd_idx_list,
            )
            generated_parameters.update(dec_adaptor_params)
        return generated_parameters

    def generate_adaptor_params(self, layer_embs, Wu_mlp, Wd_mlp,
                                Wu_name_list, Wd_name_list, Wu_idx_list, Wd_idx_list):
        """
        layer_embs: [B, num_layers, D]
        Wu_mlp, Wd_mlp: BaseMLP for up/down projection
        embedding_obj: HyperEmbeddingforAdaptor instance
        returns: dict(name -> tensor)
        """
        generated_parameters = {}
        gen_Wu = Wu_mlp(layer_embs[:, Wu_idx_list ,:])
        gen_Wd = Wd_mlp(layer_embs[:, Wd_idx_list,:])
        #print("gen_Wd : ", gen_Wd.shape) # [B, layer, 1, d_embs]
        if self.hypernet_config['parameter_efficiency_mode']:

            layer_names = []
            layer_to_indices = {}
            for i, name in enumerate(Wu_name_list):
                # e.g. "decoder_ffn_3_Wu_2" → "decoder_ffn_3"
                layer_name = "_".join(name.split("_")[:3])
                if layer_name not in layer_to_indices:
                    layer_to_indices[layer_name] = []
                    layer_names.append(layer_name) 
                layer_to_indices[layer_name].append(i)
            
            for layer_name in layer_names:
                idxs = torch.tensor(layer_to_indices[layer_name], device=gen_Wu.device, dtype=torch.long)
                Wu_tensor = gen_Wu.index_select(1, idxs)  # [B, rank, d_embs]
                Wd_tensor = gen_Wd.index_select(1, idxs)  # [B, rank, d_embs]
                generated_parameters[f"{layer_name}_Wu"] = Wu_tensor.squeeze(2)
                generated_parameters[f"{layer_name}_Wd"] = Wd_tensor.squeeze(2).transpose(1, 2)
                #print("Wu_tensor :", Wu_tensor.shape)
                #print("Wd_tensor :", Wd_tensor.shape)
        else:
            for i, name in enumerate(Wu_name_list):
                generated_parameters[name] = gen_Wu[:, i, :, :] # [B, rank, d_embs]
            for i, name in enumerate(Wd_name_list):
                generated_parameters[name] = gen_Wd[:, i, :, :] # [B, d_embs, rank]

        
        return generated_parameters


    def generate_prefix_params(self, layer_embs, prefix_key_mlp, prefix_value_mlp,
                            key_name_list, value_name_list, key_idx_list, value_idx_list):
        """
        layer_embs: [B, L, D]
        prefix_key_mlp/value_mlp: BaseMLP modules
        name_list: list of prefix token names
        idx_list: list of embedding index
        returns: dict(layer_name -> {"key": tensor, "value": tensor})
        """
        
        generated_parameters = {}

        # key prefix 
        gen_key = prefix_key_mlp(layer_embs[:,key_idx_list,:]).squeeze(-2)
        gen_val = prefix_value_mlp(layer_embs[:,value_idx_list,:]).squeeze(-2)

        layer_names = []
        layer2idx = {}
        for i, name in enumerate(key_name_list):
            layer_name = "_".join(name.split("_")[:3]) + "_prefix"
            if layer_name not in layer2idx:
                layer2idx[layer_name] = []
                layer_names.append(layer_name)
            layer2idx[layer_name].append(i)

        for layer_name in layer_names:
            idxs = torch.tensor(layer2idx[layer_name], device=gen_key.device, dtype=torch.long)
            key_tensor = gen_key.index_select(1, idxs)
            val_tensor = gen_val.index_select(1, idxs)
            generated_parameters[f"{layer_name}_key"] = key_tensor
            generated_parameters[f"{layer_name}_value"] = val_tensor

        return generated_parameters

class HyperEmbedding(nn.Module):
    def __init__(self, model_config, hypernet_config):
        super().__init__()
        self.model_config = model_config
        self.hypernet_config = hypernet_config
        self.name_to_idx = None
        self.embedding = None

    def get_embeddings(self, namelist):
        idxs = [self.name_to_idx[n] for n in namelist]
        return self.embedding.weight[idxs]
    

class HyperEmbeddingforPrefix(HyperEmbedding):
    def __init__(self, ctrl_dim, model_config, hypernet_config, block_name=None, layer_name=None):
        super().__init__(model_config, hypernet_config)
        self.block_name = block_name
        self.layer_name = layer_name

        self.name_to_idx = self.create_dictionary_name_to_idx()
        self.key_dict = {key: value for key, value in self.name_to_idx.items() if "key" in key} 
        self.value_dict = {key: value for key, value in self.name_to_idx.items() if "value" in key} 
        self.key_name_list = list(self.key_dict.keys()) 
        self.value_name_list = list(self.value_dict.keys()) 
        self.key_idx_list = list(self.key_dict.values()) 
        self.value_idx_list = list(self.value_dict.values())
        
        self.name_list = list(self.name_to_idx.keys())

        self.embedding = nn.Embedding(len(self.name_to_idx),
                                      embedding_dim=ctrl_dim)

    def create_dictionary_name_to_idx(self):
        dictionary = {}
        idx = 0
        if self.block_name == "decoder":
            for i in range(self.model_config[f'num_{self.block_name}_layers']):
                for j in range(self.hypernet_config['num_prefix']):
                    dictionary[f'{self.block_name}_{self.layer_name}_{i}_key_prefix_{j}'] = idx; idx += 1
                    dictionary[f'{self.block_name}_{self.layer_name}_{i}_value_prefix_{j}'] = idx; idx += 1
        # elif self.block_name == "encoder":
        #     num_layers =  self.model_config[f'num_{self.block_name}_layers']
        #     for i in range(num_layers//2, num_layers,1):
        #         for j in range(self.hypernet_config['num_prefix']):
        #             dictionary[f'{self.block_name}_{self.layer_name}_{i}_key_prefix_{j}'] = idx; idx += 1
        #             dictionary[f'{self.block_name}_{self.layer_name}_{i}_value_prefix_{j}'] = idx; idx += 1
        else:
            raise ValueError(f"wrong block name. input block.name : {self.block_name}")
        return dictionary


class HyperEmbeddingforAdaptor(HyperEmbedding):
    def __init__(self, ctrl_dim, model_config, hypernet_config, block_name=None, weight_type=None):
        super().__init__(model_config, hypernet_config)
        self.block_name = block_name
        self.weight_type = weight_type

        self.name_to_idx = self.create_dictionary_name_to_idx()

        self.Wu_dict = {key: value for key, value in self.name_to_idx.items() if "Wu" in key} 
        self.Wd_dict = {key: value for key, value in self.name_to_idx.items() if "Wd" in key} 
        self.Wu_name_list = list(self.Wu_dict.keys()) 
        self.Wd_name_list = list(self.Wd_dict.keys()) 
        self.Wu_idx_list = list(self.Wu_dict.values()) 
        self.Wd_idx_list = list(self.Wd_dict.values())

        self.embedding = nn.Embedding(len(self.name_to_idx),
                                      embedding_dim=ctrl_dim)

    def create_dictionary_name_to_idx(self):
        dictionary = {}
        idx = 0

        if self.block_name == "decoder":
            for i in range(self.model_config[f'num_{self.block_name}_layers']):
                if self.hypernet_config['parameter_efficiency_mode']:
                    for j in range(self.hypernet_config['rank']):
                        dictionary[f'{self.block_name}_ffn_{i}_Wu_{j}'] = idx; idx += 1
                        dictionary[f'{self.block_name}_ffn_{i}_Wd_{j}'] = idx; idx += 1
                else:
                    dictionary[f'{self.block_name}_ffn_{i}_Wu'] = idx; idx += 1
                    dictionary[f'{self.block_name}_ffn_{i}_Wd'] = idx; idx += 1
        # elif self.block_name == "encoder":
        #     num_layers =  self.model_config[f'num_{self.block_name}_layers']
        #     for i in range(num_layers//2, num_layers,1):
        #         if self.hypernet_config['parameter_efficiency_mode']:
        #             for j in range(self.hypernet_config['rank']):
        #                 dictionary[f'{self.block_name}_ffn_{i}_Wu_{j}'] = idx; idx += 1
        #                 dictionary[f'{self.block_name}_ffn_{i}_Wd_{j}'] = idx; idx += 1
        #         else:
        #             dictionary[f'{self.block_name}_ffn_{i}_Wu'] = idx; idx += 1
        #             dictionary[f'{self.block_name}_ffn_{i}_Wd'] = idx; idx += 1
        else:
            raise ValueError(f"wrong block name. input block.name : {self.block_name}")
        return dictionary

class HyperEmbeddingforInternalControl(HyperEmbedding):
    def __init__(self, ctrl_dim, model_config, hypernet_config, block_name=None, weight_type=None):
        super().__init__(model_config, hypernet_config)
        self.block_name = block_name
        self.weight_type = weight_type

        self.name_to_idx = self.create_dictionary_name_to_idx()

        self.q_dict = {key: value for key, value in self.name_to_idx.items() if "control_token" in key} 
        self.q_name_list = list(self.q_dict.keys()) 
        self.q_idx_list = list(self.q_dict.values()) 
        self.embedding = nn.Embedding(len(self.name_to_idx),
                                      embedding_dim=ctrl_dim)

    def create_dictionary_name_to_idx(self):
        dictionary = {}
        idx = 0
        for i in range(self.model_config[f'num_{self.block_name}_layers']):
            dictionary[f'{self.block_name}_control_token_layer_{i}'] = idx; idx += 1
        return dictionary

class MultiHeadAttention(nn.Module):
    def __init__(self, in_q_dim, in_kv_dim, inter_dim, out_dim, num_heads=16, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.in_q_dim = in_q_dim
        self.in_kv_dim = in_kv_dim
        self.inter_dim = inter_dim
        self.out_dim = out_dim
        self.q_proj = nn.Linear(in_q_dim, inter_dim)
        self.k_proj = nn.Linear(in_kv_dim, inter_dim)
        self.v_proj = nn.Linear(in_kv_dim, inter_dim)
        self.o_proj = nn.Linear(inter_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, q_mask=None, k_mask=None):
        """
        query : [B, Nq, Dq]
        key   : [B, Nk, Dk]
        value : [B, Nk, Dk]
        q_mask: [B, Nq]
        k_mask: [B, Nk]
        """
        if key is None:
            key = query
        if value is None:
            value = key

        B, Nq, Dq = query.shape
        _, Nk, Dk = key.shape
        H = self.num_heads
        head_dim = self.inter_dim // H

        Q = self.q_proj(query).view(B, Nq, H, head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Nk, H, head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Nk, H, head_dim).transpose(1, 2)

        # --- scaled dot-product ---
        attn_scores = (Q @ K.transpose(-2, -1)) / (head_dim ** 0.5)

        # --- key mask (block attention to invalid keys)
        if k_mask is not None:
            k_mask = k_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Nk]
            attn_scores = attn_scores.masked_fill(k_mask == 0, -1e9)

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V  # [B, H, Nq, d_h]
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.inter_dim)
        out = self.o_proj(out)

        # --- query mask (zero out invalid queries after attention)
        if q_mask is not None:
            q_mask = q_mask.unsqueeze(-1)  # [B, Nq, 1]
            out = out * q_mask

        return out
