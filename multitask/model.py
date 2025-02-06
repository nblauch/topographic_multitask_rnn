""" Network model that works with train_pytorch.py """

import torch
from torch import nn, jit
from .train import generate_trials as gen_trials
from .ei_rnn import EIRNNLayers
from .wiring_cost import accumulate_spatial_losses, SpatialLoss

class Model(nn.Module):
    def __init__(self, hp, rnn_layer): 
        super().__init__()
        n_input, n_rnn, n_output, decay, eff, num_layers = hp['n_input'], hp['n_rnn'], hp['n_output'], hp['decay'], hp['eff'], hp['num_layers']
        
        if hp['activation'] == 'relu':    # Type of activation runctions, relu, softplus, tanh, elu
            nonlinearity = nn.ReLU()
        elif hp['activation'] == 'tanh': 
            nonlinearity = nn.Tanh()
        else: 
            raise NotImplementedError
        
        self.n_rnn = n_rnn
        self.rnn   = rnn_layer(n_input, n_rnn, nonlinearity, decay, eff=eff, num_layers=num_layers)
        self.readout = nn.Linear(n_rnn, n_output, bias = False)

        self.to(hp['device'])

    def forward(self, x):
        hidden0   = torch.zeros([1, x.shape[1], self.n_rnn], device=x.device)  # initial hidden state
        hidden, _ = self.rnn(x, hidden0)
        output    = self.readout(hidden)
        return output, hidden     
    
    def get_spatial_params(self):
        return self.rnn.get_spatial_params()


class TopoModel(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        n_input, n_rnn, n_output, decay, cell_type, num_layers = hp['n_input'], hp['n_rnn'], hp['n_output'], hp['decay'], hp['rnn_type'], hp['num_layers']
        use_norm = hp['use_norm']
        nonlinearity = hp['activation']
        self.hidden_size = n_rnn
        self.rnn = EIRNNLayers(num_layers, n_input, n_rnn, nonlinearity=nonlinearity, alpha=1-decay, cell_type=cell_type, use_norm=use_norm)
        self.hid_mult = self.rnn.rnncells[0].hid_mult
        self.out_mult = self.rnn.rnncells[0].out_mult
        # simple positive-only readout, enforcing dale's law
        # self.readout = EReadout(n_rnn, n_output, bias=False)
        self.readout = nn.Linear(self.out_mult*n_rnn, n_output, bias = False)

        self.to(hp['device'])

    def forward(self, x):
        hidden0   = torch.zeros([1, x.shape[1], self.hid_mult*self.hidden_size], device=x.device)  # initial hidden state
        rnn_output, hidden = self.rnn(x, hidden0)
        output    = self.readout(rnn_output)
        return output, hidden     
    
    def get_spatial_params(self):
        return self.rnn.get_spatial_params()
    
    def enforce_connectivity(self):
        self.rnn.enforce_connectivity()

class ModelWrapper(nn.Module): #(jit.ScriptModule):
    def __init__(self, hp, model, logger=None):
        super().__init__()
        self.hp = hp
        self.model = model
        self.loss_fnc = nn.MSELoss() if hp['loss_type'] == 'lsq' else nn.CrossEntropyLoss()
        if hp['spatial_weight'] > 0:
            self.spatial_loss = SpatialLoss(weight_norm=hp['spatial_weight_norm'], dist_norm=hp['spatial_dist_norm'], style=hp['spatial_style'])
        else:
            self.spatial_loss = None
        self.logger = logger

    def generate_trials(self, rule, hp, mode, batch_size):
        return gen_trials(rule, hp, mode, batch_size)
    
    def calculate_loss(self, output, hidden, trial, hp):
        loss     = self.loss_fnc(trial.c_mask * output, trial.c_mask * trial.y)
        if not isinstance(hidden, list):
            hidden = [hidden]
        loss_reg = 0
        loss_space = 0
        for ii, h in enumerate(hidden):
            l1_activity = h.abs().mean() * hp['l1_h'] 
            l2_activity = h.norm() * hp['l2_h']  #    Regularization cost  (L1 and L2 cost) on hidden activity
            if torch.isnan(l1_activity) or torch.isnan(l2_activity):
                raise ValueError('L1 or L2 is NaN')
            loss_reg += l1_activity + l2_activity

        for param in self.parameters():
            l1 = param.abs().mean() * hp['l1_weight'] 
            l2 = param.norm() * hp['l2_weight']   #    Regularization cost  (L1 and L2 cost) on weights
            if torch.isnan(l1) or torch.isnan(l2):
                raise ValueError('L1 or L2 is NaN')
            loss_reg += l1 + l2
        
        if self.spatial_loss is not None and hasattr(self.model, 'get_spatial_params'):
            spatial_params = self.model.get_spatial_params()
            loss_space += accumulate_spatial_losses(spatial_params, self.spatial_loss) #    Spatial regularization cost on weights

        return loss, loss_reg, loss_space
    
#     @jit.script_method
    def forward(self, rule, batch_size = None, mode = 'random'): #, **kwargs):
        hp             = self.hp        
        trial          = self.generate_trials(rule, hp, mode, batch_size)
        output, hidden = self.model(trial.x)
        loss, loss_reg, loss_space = self.calculate_loss(output, hidden, trial, hp)
        if not isinstance(hidden, list):
            hidden = [hidden]
        output = output.detach().cpu().numpy()
        hidden = [h.detach().cpu().numpy() for h in hidden]
        return loss, loss_reg, loss_space, output, hidden, trial
    
    