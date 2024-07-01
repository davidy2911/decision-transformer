"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        
        # self.register_buffer("mask", (1 - torch.tril(torch.ones(config.block_size + 1, config.block_size + 1)))
        #                              .view(1, 1, config.block_size + 1, config.block_size + 1).transpose(2,3))
        self.n_head = config.n_head

        print(config.n_embd, config.block_size)

    def forward(self, x, layer_past=None):
        # print(x.size())
        # print(self.n_head, self.mask.shape)
        # exit()
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.head = nn.Sequential(
            # nn.LayerNorm(config.n_emb),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(config.n_embd, config.vocab_size),
            # nn.Tanh() 
        )

        self.pred_rating = nn.Linear(config.n_embd, 1, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # self.pos_emb = nn.Embedding(config.max_timestep, config.n_embd)
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        # self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Linear(config.n_embd, config.n_embd, bias=False), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())

        self.state_pred = nn.Sequential(nn.Linear(config.n_embd, config.n_embd, bias=False), nn.Tanh())

        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        # self.predict_return = torch.nn.Linear(config.n_embd, 1)


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    # def forward(self, actions, targets=None, ratings=None, timesteps=None, attention=None):
    #     # states: (batch, block_size, 4*84*84)
    #     # actions: (batch, block_size, 1)
    #     # targets: (batch, block_size, 1)
    #     # rtgs: (batch, block_size, 1)
    #     # timesteps: (batch, block_size, 1)


    #     # print(actions.shape, timesteps.shape)
    #     B, T = actions.size()
    #     rating_embeddings = self.ret_emb(ratings.type(torch.float32))
    #     action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
    #     time_embeddings = self.pos_emb(timesteps.type(torch.long).squeeze(-1))

    #     action_embeddings = action_embeddings + time_embeddings
    #     rating_embeddings = rating_embeddings + time_embeddings


    #     token_embeddings = torch.zeros((B, T*2 , self.config.n_embd), dtype=torch.float32, device=action_embeddings.device)


    #     # token_embeddings[:,::2,:] = action_embeddings # really just [:,0,:]
    #     # token_embeddings[:,1::2,:] = rating_embeddings # really just [:,1,:]

    #     # tout = self.drop(token_embeddings)
    #     # x = self.blocks(tout)
    #     # x = self.ln_f(x)
    #     # # logits = self.head(x)

    #     # # acts = x[:, ::2, :]
    #     # preds = self.pred_rating(x[:,1::2,:])

    #     # loss = torch.mean((ratings.reshape(-1) - preds.reshape(-1))**2)


    #     token_embeddings[:,::2,:] = rating_embeddings # really just [:,0,:]
    #     token_embeddings[:,1::2,:] = action_embeddings # really just [:,1,:]

    #     tout = self.drop(token_embeddings)
    #     x = self.blocks(tout)
    #     x = self.ln_f(x)
    #     preds = self.head(x[:, ::2, :])

    #     loss = None 
    #     if targets is not None:
    #         loss = F.cross_entropy(preds.reshape(-1, preds.size(-1)), targets.reshape(-1), ignore_index=-1, size_average=True)

    #     # if actions is not None and self.model_type == 'reward_conditioned':
    #     #     logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
    #     # elif actions is None and self.model_type == 'reward_conditioned':
    #     #     logits = logits[:, 1:, :]
    #     # elif actions is not None and self.model_type == 'naive':
    #     #     logits = logits[:, ::2, :] # only keep predictions from state_embeddings
    #     # elif actions is None and self.model_type == 'naive':
    #     #     logits = logits # for completeness
    #     # else:
    #     #     raise NotImplementedError()


    #     return preds, loss
    

    def forward(self, actions, targets=None, ratings=None, timesteps=None, attention=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        batch_size, token_len = actions.size()
        rating_embeddings = self.ret_emb(ratings.type(torch.float32))
        action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
        # time_embeddings = self.pos_emb(timesteps.type(torch.long).squeeze(-1))

        # action_embeddings = action_embeddings + time_embeddings
        # rating_embeddings = rating_embeddings + time_embeddings


        token_embeddings = torch.zeros((batch_size, token_len*2 , self.config.n_embd), dtype=torch.float32, device=action_embeddings.device)

        token_embeddings[:,::2,:] = rating_embeddings # really just [:,0,:]
        token_embeddings[:,1::2,:] = action_embeddings # really just [:,1,:]
        
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd

        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        state_embd = self.state_pred(x[:, 1::2, :])
        # print(state_embd.shape, action_embeddings.shape)

        actions_preds = self.head(x[:, ::2, :])

        loss = None
        rating_loss = None
        action_loss = None
        if targets is not None:
            # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            # pred_loss = torch.mean((pred_returns - rtgs)**2)

            # pred = torch.sum(state_embd[:, :-1, :] * action_embeddings[:, 1:, :], dim=-1)
            # print(pred.shape, ratings.shape)
            rating_loss = torch.mean(( torch.sum(state_embd[:, :-1, :] * action_embeddings[:, 1:, :], dim=-1) - torch.squeeze(ratings[:, 1:, :]))**2) 
            action_loss = F.cross_entropy(actions_preds.reshape(-1, actions_preds.size(-1)), targets.reshape(-1), ignore_index=-1, size_average=True) #+ torch.mean((pred_returns - rtgs)**2)

            # loss = action_loss + rating_loss

            # actions_true = F.one_hot(targets, num_classes=self.config.vocab_size)
            # loss = torch.mean((logits - actions_true)**2)
        # print(logits.shape, actions.shape)

        return actions_preds, action_loss, rating_loss
    
    def get_action(self, actions, ratings=None, timesteps=None, attention=None, device='cpu'):
        # we don't care about the past rewards in this model
        # print(actions)
        # print(returns_to_go)

        with torch.no_grad():
            actions = torch.from_numpy(np.array(actions).reshape(1, -1)).long().to(device)
            ratings = torch.from_numpy(np.array(ratings).reshape(1, -1, 1)).float().to(device)
            # timesteps = torch.from_numpy(np.array(timesteps).reshape(1, -1)).long().to(device)

            timesteps = torch.from_numpy(np.array(timesteps).reshape(1, -1, 1)).long().to(device)
            # mask = torch.ones(ratings.shape[1]).reshape(1, -1).long().to(device)
            action_preds , _ , _ = self.forward(
                actions=actions,
                targets=None,
                ratings=ratings,
                timesteps=timesteps,
                attention=None
            )


        
        # print(action_preds.shape)
        # exit()
        return action_preds
    

    
