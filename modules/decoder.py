import torch
import torch.nn as nn

from utils.train_utils import weight_init, get_causal_mask
from utils.layers import MLPLayer, ResidualMLP
import torch.nn.functional as F 

class Decoder(nn.Module):

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_rl_waymo = self.cfg.dataset.waymo
        self.use_k_actions = getattr(self.cfg_rl_waymo, 'action_space', 'grid') == 'k_action'
        self.use_residual_mlp = getattr(self.cfg_model, 'use_residual_mlp', False)
        self.residual_layers = getattr(self.cfg_model, 'residual_mlp_layers', 2)
        self.grid_action_dim = self.cfg_rl_waymo.accel_discretization * self.cfg_rl_waymo.steer_discretization
        self.k_action_dim = getattr(self.cfg_rl_waymo, 'k_action_vocab_size', None)
        if self.use_k_actions and self.k_action_dim is None:
            raise ValueError("cfg.dataset.waymo.k_action_vocab_size must be set when action_space is k_action.")
        self.action_dim = self.k_action_dim if self.use_k_actions else self.grid_action_dim
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.cfg_model.hidden_dim, 
                                                                                    dim_feedforward=self.cfg_model.dim_feedforward,
                                                                                    nhead=self.cfg_model.num_heads,
                                                                                    batch_first=True), 
                                                                                    num_layers=self.cfg_model.num_decoder_layers)
        self.predict_grid_action = self._build_mlp(self.cfg_model.hidden_dim, self.grid_action_dim)
        self.predict_k_action = self._build_mlp(self.cfg_model.hidden_dim, self.k_action_dim) if self.k_action_dim is not None else None

        if self.cfg_model.predict_rtg:
            self.predict_rtg = self._build_mlp(self.cfg_model.hidden_dim, self.cfg_rl_waymo.rtg_discretization * self.cfg_model.num_reward_components)

        if self.cfg_model.predict_future_states:
            self.predict_future_states = self._build_mlp(self.cfg_model.hidden_dim, self.cfg_rl_waymo.train_context_length * 2, hidden_dim=self.cfg_model.hidden_dim)

        if not (self.cfg_model.trajeglish or self.cfg_model.il):
            num_types = 3
        elif self.cfg_model.trajeglish:
            num_types = 1
        else:
            num_types = 2
        self.causal_mask = get_causal_mask(self.cfg, self.cfg_rl_waymo.train_context_length, num_types)
        self.apply(weight_init)


    def _build_mlp(self, input_dim, output_dim, hidden_dim=None):
        hidden_dim = hidden_dim or self.cfg_model.hidden_dim
        if self.use_residual_mlp:
            return ResidualMLP(input_dim=input_dim,
                               hidden_dim=hidden_dim,
                               n_hidden=self.residual_layers,
                               output_dim=output_dim)
        return MLPLayer(input_dim, hidden_dim, output_dim)


    def _reshape_actions(self, logits, batch_size, seq_len, action_dim):
        if logits is None:
            return None
        return logits.reshape(batch_size, seq_len, self.cfg_rl_waymo.max_num_agents, action_dim).permute(0, 2, 1, 3)


    def forward(self, data, scene_enc, eval=False):
        agent_states = data['agent'].agent_states
        batch_size = agent_states.shape[0]
        seq_len = agent_states.shape[2]
        
        # [batch_size, num_timesteps * num_agents * 3, hidden_dim]
        stacked_embeddings = scene_enc['stacked_embeddings']
        # [batch_size, num_polyline_tokens + num_initial_state_tokens, hidden_dim]
        encoder_embeddings = scene_enc['encoder_embeddings']
        # [batch_size, num_polyline_tokens + num_initial_state_tokens]
        src_key_padding_mask = scene_enc['src_key_padding_mask']
        num_timesteps = agent_states.shape[2]
        
        output = self.transformer_decoder(stacked_embeddings, encoder_embeddings, tgt_mask=self.causal_mask.to(stacked_embeddings.device), memory_key_padding_mask=src_key_padding_mask)
        
        preds = {}
        if not (self.cfg_model.trajeglish or self.cfg_model.il):
            # [batch_size, 3, num_timesteps * num_agents, hidden_dim]
            output = output.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, 3, self.cfg_model.hidden_dim).permute(0, 2, 1, 3)
            action_token = output[:, 1]
        elif self.cfg_model.trajeglish:
            output = output.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, 1, self.cfg_model.hidden_dim).permute(0, 2, 1, 3)
            action_token = output[:, 0]
        else:
            output = output.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, 2, self.cfg_model.hidden_dim).permute(0, 2, 1, 3)
            action_token = output[:, 0]

        grid_action_logits = self.predict_grid_action(action_token) if self.predict_grid_action is not None else None
        k_action_logits = self.predict_k_action(action_token) if self.predict_k_action is not None else None
        grid_action_preds = self._reshape_actions(grid_action_logits, batch_size, seq_len, self.grid_action_dim)
        k_action_preds = self._reshape_actions(k_action_logits, batch_size, seq_len, self.k_action_dim) if self.predict_k_action is not None else None

        # Collect prediction outputs
        if grid_action_preds is not None:
            preds['grid_action_preds'] = grid_action_preds
        if k_action_preds is not None:
            preds['k_action_preds'] = k_action_preds
        
        # Final action predictions based on action space 
        if self.use_k_actions:
            if k_action_preds is None:
                raise ValueError("k_action head is not initialized but action_space is k_action.")
            preds['action_preds'] = k_action_preds
        else:
            preds['action_preds'] = grid_action_preds

        if self.cfg_model.predict_future_states:
            state_preds = self.predict_future_states(output[:, 2])
            state_preds = state_preds.reshape(batch_size, seq_len, self.cfg_rl_waymo.max_num_agents, self.cfg_rl_waymo.train_context_length * 2).permute(0, 2, 1, 3)
            preds['state_preds'] = state_preds
        
        if self.cfg_model.predict_rtg:
            rtg_preds = self.predict_rtg(output[:, 0])
            rtg_preds = rtg_preds.reshape(batch_size, seq_len, self.cfg_rl_waymo.max_num_agents, self.cfg_rl_waymo.rtg_discretization * self.cfg_model.num_reward_components).permute(0, 2, 1, 3)
            preds['rtg_preds'] = rtg_preds

        return preds
