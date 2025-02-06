import torch
from torch import nn
import numpy as np
import scipy.spatial

class EIRNNCellBase(nn.Module):
    """
    Abstract base class for EIRNNCell, EIEFFRNNCell, RNNCell

    Does not define the init function, which must be defined by the subclass
    """
    def forward(self, inputs, state, preact_state):

        # initialize states if first time step, with separate e and i populations
        if state == None:
            state = torch.zeros((inputs.shape[0], self.hid_mult*self.hid_size), device=inputs.device)
        if preact_state == None:
            preact_state = torch.zeros((inputs.shape[0], self.hid_mult*self.hid_size), device=inputs.device)

        # compute multiplicative noise on weights
        ih_weight, hh_weight = self.get_connectivity()

        # compute preactivation
        new_preact_state = inputs.matmul(ih_weight.t()) + state.matmul(hh_weight.t())

        # compute additive noise on preactivation
        new_preact_state = apply_noise(new_preact_state, self.act_noise)

        # do layer normalization and add bias since we turned LN bias off
        new_preact_state = self.norm(new_preact_state) + self.bias

        # integrate state according to time constants and apply nonlinearity
        # state = self.nonlin((1-self.ei_alphas)*preact_state + self.ei_alphas*new_preact_state)

        # this appears to work better in the Yang tasks
        state = state*(1-self.ei_alphas) + self.nonlin(new_preact_state)*self.ei_alphas

        return state, state, new_preact_state

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=np.sqrt(5))    #, nonlinearity=nonlinearity)
        nn.init.kaiming_uniform_(self.weight_hh, a=np.sqrt(5))    #, nonlinearity=nonlinearity)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self._clamp_positive(init=True)

    def get_norm(self):
        if self.use_norm:
            return nn.LayerNorm(self.hid_mult*self.hid_size, elementwise_affine=False)
        else:
            return nn.Identity()
        
    def get_coords(self, coords=None):
        if coords is None:
            assert np.mod(np.sqrt(self.hid_size), 1) == 0, 'hid_size must be a perfect square'
            X, Y = np.meshgrid(np.linspace(0, 1, int(np.sqrt(self.hid_size))), np.linspace(0, 1, int(np.sqrt(self.hid_size))))
            coords = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
            # arrange e and i neurons in the same location (functional columns)
            coords = np.concatenate([coords for ii in range(self.hid_mult)], axis=0)
        else:
            assert coords.shape[0] == self.hid_mult*self.hid_size
        return coords

    def _clamp_positive(self, init=False):
        """
        the eimask's control the sign, so the weights must always be positive

        this should be called after optimizer.step(), to ensure that the next minibatch has correct weights

        we use abs at init to have a denser representation, and clamp everywhere else for stabler learning
        """
        if init:
            if self.eimask_ih is not None:
                self.weight_ih.data = torch.abs(self.weight_ih.data)
            if self.eimask_hh is not None:
                self.weight_hh.data = torch.abs(self.weight_hh.data)
        else:
            if self.eimask_ih is not None:
                self.weight_ih.data = torch.clamp(self.weight_ih.data, min=0)
            if self.eimask_hh is not None:
                self.weight_hh.data = torch.clamp(self.weight_hh.data, min=0)  
    
    def get_connectivity(self):
        """
        combine learned weight and fixed EI mask into a single weight for IH and HH
        """ 
        if self.eimask_ih is not None:
            ih_weight = torch.clamp(self.weight_ih, min=0)*self.eimask_ih
        else:
            ih_weight = self.weight_ih
        if self.eimask_hh is not None:
            hh_weight = torch.clamp(self.weight_hh, min=0)*self.eimask_hh
        else:
            hh_weight = self.weight_hh

        return ih_weight, hh_weight    

    def _set_connection_distances(self, *args, **kwargs):
        """
        helper function to set the connection distances, utilizes _get_connection_distances
        """
        if self.input_coords is not None:
            self.ff_distances = self._get_connection_distances(self.input_coords, self.coords, *args, **kwargs)
        else:
            self.ff_distances = torch.zeros(self.hid_mult*self.hid_size, self.in_size).to(self.device)
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

        D = D.transpose() # (receiving, sending)

        return torch.tensor(D, device=device)
    
    def get_spatial_params(self):
        """
        return list of (weight, distance, conn_type) tuples for computing wiring costs
        """
        return [
            (self.weight_ih, self.ff_distances, 'ff'),
            (self.weight_hh, self.rec_distances, 'rec'),
        ]
    
    def reshape_units(self, unit_vector):
        sl = int(np.sqrt(self.hid_size))
        e_units = unit_vector[:self.hid_size].reshape(sl, sl)
        if unit_vector.shape[0] > self.hid_size:
            i_units = unit_vector[self.hid_size:].reshape(sl, sl)
            return np.concatenate([e_units, i_units], 0)
        else:
            return e_units


class EIRNNCell(EIRNNCellBase):
    """
    A recurrent cell with separate populations of excitatory and inhibitory units with recurrent connections to each other.
    Each cell type connects to both cell types, and both receive driving feedforward inputs. 
    """
    def __init__(self, in_size, hid_size, nonlinearity='sigmoid', bias=True, device='cuda', use_norm=True, alpha=0.2, ei_alpha_ratio=1,
                 act_noise=0, wrap_x=True, wrap_y=True, coords=None, input_coords=None,
                 ):
        super().__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.device = device
        self.wrap_x = wrap_x
        self.wrap_y = wrap_y
        self.hid_mult = 2 # number of hidden cell types
        self.out_mult = 2 # number of hidden cell types sending outputs
        self.use_norm = use_norm
        self.act_noise = act_noise # sd of noise on preactivation
        self.weight_ih = nn.Parameter(torch.Tensor(2*hid_size, in_size)) # E+I hidden units, E inputs
        self.weight_hh = nn.Parameter(torch.Tensor(2*hid_size, 2*hid_size))
        self.nonlin = get_nonlin(nonlinearity)

        # alpha is the unitless time constant (dt/tau). higher alpha means faster integration
        assert alpha is not None
        assert ei_alpha_ratio is not None
        self.alpha = alpha
        # ei_alpha ratio is the ratio of the time constants of the E and I neurons
        self.ei_alpha_ratio = ei_alpha_ratio # alpha_e = alpha*ei_alpha_ratio, alpha_i = alpha
        self.ei_alphas = torch.zeros((2*hid_size,), device=device)
        self.ei_alphas[:hid_size] = alpha*ei_alpha_ratio # E neurons
        self.ei_alphas[hid_size::] = alpha # I neurons

        # set up masks to constrain E/I unit signs
        self.eimask_ih = torch.zeros((2*hid_size, in_size), device=device)
        self.eimask_hh = torch.zeros((2*hid_size, 2*hid_size), device=device)
        with torch.no_grad():
            for mask in [self.eimask_hh, self.eimask_ih]:
                mask[:, :hid_size] = 1 # positive recurrent weights from E to E and I
                mask[:, hid_size::] = -1 # negative recurrent weights from I to E and I
                # no self-connections (per Song, Yang, Wang)
                mask.fill_diagonal_(0)
        if input_coords is None:
            self.eimask_ih = None

        self.norm = self.get_norm()

        if bias:
            self.bias = nn.Parameter(torch.Tensor(2*hid_size))
        else:
            self.register_parameter('bias', 0, requires_grad=False)

        self.to(device)
        self.reset_parameters()

        self.coords = self.get_coords(coords)

        self.input_coords = input_coords

        self._set_connection_distances(wrap_x, wrap_y, device=device)
    
class EIEFFRNNCell(EIRNNCell):
    """
    A recurrent cell with separate populations of excitatory and inhibitory units with recurrent connections to each other.
    Each cell type connects to both cell types, and both receive driving feedforward inputs. Only E cells send feedforward outputs.
    The main IT motif used in the ITN of Blauch et. al, 2022 PNAS
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # positive feedforward weights from E units only
        if self.input_coords is not None:
            self.eimask_ih = torch.ones((2*self.hid_size, self.in_size), device=self.device) 
            self.ff_distances = self.ff_distances[:,:self.in_size]
        self.out_mult = 1
        self.reset_parameters()

    def forward(self, *args, **kwargs):
        """
        return only the excitatory output
        """
        _, state, preact_state = super().forward(*args, **kwargs)
        outputs = state[:,:self.hid_size]
        return outputs, state, preact_state
    
class RNNCell(EIRNNCellBase):
    """
    A simple RNN without sign constraints
    """
    def __init__(self, in_size, hid_size, nonlinearity='sigmoid', bias=True, device='cuda', use_norm=True, alpha=0.2, ei_alpha_ratio=1,
                 act_noise=0, wrap_x=True, wrap_y=True, coords=None, input_coords=None,
                 ):
        super().__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.device = device
        self.wrap_x = wrap_x
        self.wrap_y = wrap_y
        self.hid_mult = 1 # number of hidden cell types
        self.out_mult = 1 # number of hidden cell types sending outputs
        self.use_norm = use_norm
        self.act_noise = act_noise # sd of noise on preactivation
        self.weight_ih = nn.Parameter(torch.Tensor(self.hid_mult*self.hid_size, in_size)) # E+I hidden units, E inputs
        self.weight_hh = nn.Parameter(torch.Tensor(self.hid_mult*self.hid_size, self.hid_mult*self.hid_size))
        self.nonlin = get_nonlin(nonlinearity)

        print(f'alpha: {alpha}')

        # alpha is the unitless time constant (dt/tau). higher alpha means faster integration
        assert alpha is not None
        assert ei_alpha_ratio is None or ei_alpha_ratio == 1
        self.alpha = alpha
        self.ei_alphas = alpha
        self.ei_alpha_ratio = 1

        # set up masks to constrain E/I unit signs
        self.eimask_ih = None
        self.eimask_hh = None

        self.norm = self.get_norm()

        if bias:
            self.bias = nn.Parameter(torch.Tensor(hid_size))
        else:
            self.register_parameter('bias', 0, requires_grad=False)

        self.to(device)

        self.coords = self.get_coords(coords)

        if input_coords is not None:
            assert input_coords.shape[0] == in_size, input_coords.shape
        self.input_coords = input_coords

        self._set_connection_distances(wrap_x, wrap_y, device=device)

        self.reset_parameters()


class EFFRNNCell(RNNCell):
    """
    A simple non-Dale RNN that is constrained to have only excitatory feedforward weights (see Blauch et. al, 2022 PNAS simplified models)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.input_coords is not None:
            self.eimask_ih = torch.ones((self.hid_mult*self.hid_size, self.in_size), device=self.device)
        self.reset_parameters()
        

class EIRNNLayers(nn.Module):
    """
    Container for multiple RNN layers capable of processing input over many time steps
    """
    def __init__(self, num_layers, input_size, hidden_size, cell_type='EIEFFRNN', **kwargs):
        super().__init__()
        cells = []
        input_coords = None
        cell_type = eval(f'{cell_type}Cell')
        for _ in range(num_layers):
            cells.append(cell_type(input_size, hidden_size, input_coords=input_coords, coords=None, **kwargs))
            input_size = hidden_size*cells[-1].out_mult
            input_coords = cells[-1].coords
        self.rnncells = nn.ModuleList(cells)
    
    def forward(self, inputs, hidden_init):
        inputs = inputs.unbind(0)     # inputs has dimension [Time, batch, n_input]
        hiddens = [hidden_init[0] for _ in range(len(self.rnncells))]       # initial state has dimension [1, batch, n_input]
        all_hiddens = [ [h] for h in hiddens] # store each layer's hidden state over time
        preact_states = [None for _ in range(len(self.rnncells))]
        outputs = []
        for i in range(len(inputs)):  # looping over the time dimension 
            cell_input = inputs[i]
            for j, cell in enumerate(self.rnncells):
                output, hidden, preact_state = cell(cell_input, hiddens[j], preact_states[j])
                cell_input = output
                hiddens[j] = hidden
                all_hiddens[j].append(hidden)
                preact_states[j] = preact_state
            outputs += [output]
        all_hiddens = [torch.stack(h) for h in all_hiddens]
        return torch.stack(outputs), all_hiddens
    
    def enforce_connectivity(self):
        for cell in self.rnncells:
            cell._clamp_positive()

    def get_spatial_params(self):
        params = []
        for cell in self.rnncells:
            params.extend(cell.get_spatial_params())
        return params


def get_nonlin(nonlin_str):
    """
    simple utility toget a nonlinearity from a string
    """
    if nonlin_str.lower() == 'sigmoid':
        nonlin = nn.Sigmoid()
    elif nonlin_str.lower() == 'relu':
        nonlin = nn.ReLU()
    elif nonlin_str.lower() == 'relu6':
        nonlin = nn.ReLU6()
    elif nonlin_str.lower() == 'tanh':
        nonlin = nn.Tanh()
    else:
        raise NotImplementedError()
    return nonlin

def apply_noise(x, sd):
    """
    apply gaussian noise with standard deviation sd
    """
    x = x + sd*torch.randn_like(x)
    return x

def gaussian_noise(x, sd):
    """
    get gaussian noise with standard deviation sd
    """
    return sd*torch.randn_like(x)

class EReadout(nn.Linear):
    """
    A linear readout that is constrained to have only excitatory weights
    """
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self._clamp_positive(init=True)

    def forward(self, inputs):
        bias = self.bias if self.bias is not None else 0
        return inputs @ torch.clamp(self.weight, min=0).t() + bias

    def _clamp_positive(self, init=False):
        if init:
            self.weight.data = torch.abs(self.weight.data)
        else:
            self.weight.data = torch.clamp(self.weight.data, min=0)