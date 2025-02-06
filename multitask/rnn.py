""" Custom RNN implementation """

import torch
from torch import nn, jit
import math
import scipy
import numpy as np

class RNNCell_base(jit.ScriptModule):     # (nn.Module):
#     __constants__ = ['bias']
    
    def __init__(self, input_size, hidden_size, nonlinearity, bias, coords=None, input_coords=None, device='cuda'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.device = device

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size)) 
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

        if coords is None:
            assert np.mod(np.sqrt(hidden_size), 1) == 0, 'hidden_size must be a perfect square'
            X, Y = np.meshgrid(np.linspace(0, 1, int(np.sqrt(hidden_size))), np.linspace(0, 1, int(np.sqrt(hidden_size))))
            coords = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
        else:
            assert coords.shape[0] == hidden_size
        self.coords = coords

        if input_coords is not None:
            assert input_coords.shape[0] == input_size, input_coords.shape
        self.input_coords = input_coords

        self._set_connection_distances(wrap_x=True, wrap_y=True)
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))    #, nonlinearity=nonlinearity)
        nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))    #, nonlinearity=nonlinearity)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _set_connection_distances(self, *args, **kwargs):
        """
        helper function to set the connection distances, utilizes _get_connection_distances
        """
        if self.input_coords is not None:
            self.ff_distances = self._get_connection_distances(self.input_coords, self.coords, *args, **kwargs)
        else:
            self.ff_distances = torch.zeros(self.hidden_size, self.input_size).to(self.device)
        self.rec_distances = self._get_connection_distances(self.coords, self.coords, *args, **kwargs)

    def _get_connection_distances(self, from_coords, to_coords, wrap_x, wrap_y, norm=2, ignore_x=False, ignore_y=False, device='cuda'):
        """
        returns euclidean connection distances between from_coords and to_coords, subject to potential wrapping in x and y dimensions
        """
       # dimensions must be normalized to be in 0,1, but the max-min coord should be nonzero for correct wrapping
        dx = scipy.spatial.distance_matrix(np.expand_dims(from_coords[:,0],1), np.expand_dims(to_coords[:,0],1), p=1)
        dy = scipy.spatial.distance_matrix(np.expand_dims(from_coords[:,1],1), np.expand_dims(to_coords[:,1],1), p=1)
        if to_coords.shape[1] == 3:
            dz = scipy.spatial.distance_matrix(np.expand_dims(from_coords[:,2],1), np.expand_dims(to_coords[:,2],1), p=1)
        else:
            dz = 0
        if wrap_x or wrap_y:
            wrapx = dx > .5
            wrapy = dy > .5
            if wrap_x: # probably false for log polar mapping
                dx[wrapx] = 1 - dx[wrapx]
            if wrap_y:
                dy[wrapy] = 1 - dy[wrapy]
        if ignore_x:
            print('ignoring x (col index of map)')
            dx = np.zeros_like(dx)
        if ignore_y:
            print('ignoring y (row index of map)')
            dy = np.zeros_like(dy)
        D = (dx**norm + dy**norm + dz**norm)**(1/norm)
        D = D.transpose(0,1)

        return torch.tensor(D, device=device)
    
    def get_spatial_params(self):
        """
        return list of (weight, distance, conn_type) tuples for computing wiring costs
        """
        return [
            (self.weight_ih, self.ff_distances, 'ff'),
            (self.weight_hh, self.rec_distances, 'rec'),
        ]

    def get_connectivity(self):
        return self.weight_ih, self.weight_hh

            
class RNNCell(RNNCell_base):  # Euler integration of rate-neuron network dynamics 
    def __init__(self, input_size, hidden_size, nonlinearity = None, decay = 0.9, bias = True, **kwargs):
        super().__init__(input_size, hidden_size, nonlinearity, bias, **kwargs)
        self.decay = decay    #  torch.exp( - dt/tau )
        self.alpha = 1-self.decay
        print(f'alpha: {1-self.decay}')

    def forward(self, input, hidden): 
        ih_weight, hh_weight = self.get_connectivity()                       
        activity = self.nonlinearity(input @ ih_weight.t() +  hidden @ hh_weight.t() + self.bias)
        hidden   = self.decay * hidden + (1 - self.decay) * activity
        return hidden
    

class RNNLayer(nn.Module):    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rnncell = RNNCell(*args, **kwargs)

    def forward(self, input, hidden_init):
        inputs = input.unbind(0)     # inputs has dimension [Time, batch, n_input]
        hidden = hidden_init[0]      # initial state has dimension [1, batch, n_input]
        outputs = []
        for i in range(len(inputs)):  # looping over the time dimension 
            hidden = self.rnncell(inputs[i], hidden)
            outputs += [hidden]       # vanilla RNN directly outputs the hidden state
        return torch.stack(outputs), hidden
    
    def get_spatial_params(self):
        return self.rnncell.get_spatial_params()
    
class RNNLayers(nn.Module):
    def __init__(self, input_size, hidden_size, *args, num_layers=1, **kwargs):
        super().__init__()
        cells = []
        input_coords = None
        for _ in range(num_layers):
            cells.append(RNNCell(input_size, hidden_size, *args, input_coords=input_coords, **kwargs))
            input_size = hidden_size
            input_coords = cells[-1].coords[:hidden_size] # E cells only
        self.rnncells = nn.ModuleList(cells)

    def forward(self, inputs, hidden_init):
        inputs = inputs.unbind(0)     # inputs has dimension [Time, batch, n_input]
        hiddens = [hidden_init[0] for _ in range(len(self.rnncells))]       # initial state has dimension [1, batch, n_input]
        all_hiddens = [ [h] for h in hiddens] # store each layer's hidden state over time
        outputs = []
        for i in range(len(inputs)):  # looping over the time dimension 
            cell_input = inputs[i]
            for j, cell in enumerate(self.rnncells):
                hidden = cell(cell_input, hiddens[j])
                cell_input = hidden
                hiddens[j] = hidden
                all_hiddens[j].append(hidden)
            outputs += [hidden]
        all_hiddens = [torch.stack(h) for h in all_hiddens]
        return torch.stack(outputs), all_hiddens


    
    def get_spatial_params(self):
        all_spatial_params = []
        for cell in self.rnncells:
            all_spatial_params.extend(cell.get_spatial_params())
        return all_spatial_params