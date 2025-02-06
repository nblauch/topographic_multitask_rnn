from torch import nn
import torch

class SpatialLoss(nn.Module):
    def __init__(self, weight_norm=2, dist_norm=2, bias=1, style='jj'):
        super().__init__()
        assert weight_norm > 0 and dist_norm > 0
        assert style in ['jj', 'jju', 'norm', 'dot', 'sernn', 'sernn2']
        self.weight_norm = weight_norm
        self.dist_norm = dist_norm
        self.style = style
        self.bias = bias
        
    def forward(self, weights, dists, conn_type=None):
        if self.style == 'jj':
            # sum the individual weight contributions, as in Jacobs & Jordan, 1992
            loss = torch.sum((torch.abs(weights)**self.weight_norm * dists**self.dist_norm)/(self.bias+ torch.abs(weights)**self.weight_norm))
        elif self.style == 'jju':
            # similar to jj, but for unit-vector-length weights per hidden unit so that simply decreasing the weights uniformly will not decrease the loss, and all units are encouraged to participate
            weights_unit = nn.functional.normalize(weights, p=2, dim=1)
            loss = torch.sum((torch.abs(weights_unit)**self.weight_norm * dists**self.dist_norm)/(self.bias+ torch.abs(weights_unit)**self.weight_norm))
        elif self.style == 'norm':
            # use the equivalent of a distance p-norm scaled (and p-norm generalized) weight decay, normalized by the weight norm
            # note: similar to jju, simply decreasing the weights uniformly will not decrease the loss, however this doesn't explicitly encourage all units to participate
            w_pow = torch.abs(weights)**self.weight_norm
            loss = torch.sum((w_pow * dists**self.dist_norm))/(torch.sum(w_pow))
        elif self.style == 'dot':
            # use the equivalent of a distance p-norm scaled (and p-norm generalized) weight decay
            w_pow = torch.abs(weights)**self.weight_norm
            loss = torch.sum((w_pow * dists**self.dist_norm))
        elif self.style == 'sernn':
            # spatially embedded RNN cost taking into account network communicability (Achterberg et. al, 2022)
            # for feedforward weights, communicability does not make sense, so we drop the term to converge on essentially the dot style
            if conn_type == 'ff':
                c_w = torch.abs(weights)**self.weight_norm
            else:
                # calculate communicability
                eps = 1e-6
                s = torch.diag(torch.pow(torch.sum(torch.abs(weights), dim=1) + eps, -0.5))
                c = torch.linalg.matrix_exp(s@torch.abs(weights))
                c = c.fill_diagonal_(0)
                # combine communicability with weights
                c_w = torch.mul(c, torch.abs(weights)**self.weight_norm)
            loss = torch.sum(torch.mul(c_w, dists**self.dist_norm))
        elif self.style == 'sernn2':
            # a variant of the seRNN where we only use communicability
            # for feedforward weights, communicability does not make sense, so set to zero to compute zero loss
            if conn_type == 'ff':
                c = 0
            else:
                # calculate communicability
                eps = 1e-6
                s = torch.diag(torch.pow(torch.sum(torch.abs(weights), dim=1) + eps, -0.5))
                c = torch.linalg.matrix_exp(s@torch.abs(weights))
                c = c.fill_diagonal_(0)
            loss = torch.sum(torch.mul(c**self.weight_norm, dists**self.dist_norm))
        else:
            raise ValueError()
        return loss 
    
    def forward_oneweight(self, weight, dist, weights):
        """
        view the loss contribution of a single weight. just for visualization really
        """
        if self.style == 'jj':
            # sum the individual weight contributions, as in Jacobs & Jordan, 1992
            loss = self.forward(weight, dist)
        elif self.style == 'norm':
            # use the equivalent of a distance p-norm scaled (p-norm generalized) weight decay
            loss = torch.sum((torch.abs(weight)**self.weight_norm * dist**self.dist_norm))/(torch.sum(torch.abs(weights)**self.weight_norm))
        elif self.style == 'dot':
            loss = self.forward(weight, dist)
        else:
            raise ValueError()
        return loss            


def accumulate_spatial_losses(spatial_params, spatial_loss, f2rratio=1):
    wiring_cost = 0
    for param, dists, conn_type in spatial_params:
        # controllable ratio of feedforward to recurrent spatial costs
        conn_mult = f2rratio if conn_type == 'ff' else 1 
        if param.shape[1] != dists.shape[1]:
            # includes a skip connection, for whom we will not model distance
            param = param[:,:dists.shape[1]]
        this_cost = conn_mult * spatial_loss(param, dists, conn_type=conn_type)
        if torch.isnan(this_cost):
            raise ValueError('Spatial loss is NaN')
        wiring_cost += this_cost
    return wiring_cost