"""
PyTorch/Pyro Implementation of Recurrent Flow Networks 
------------------------------------------------------

This file contains the model specifications, a full GitHub repository is currently under
development.

Requirements:
- torch==1.4.0
- pyro-ppl==1.3.1

"""
import pyro
import torch
from torch import nn
from pyro.nn.dense_nn import ConditionalDenseNN
import pyro.distributions as dist
from pyro.distributions.transforms import permute, BatchNorm
from pyro.distributions.transforms.affine_coupling import ConditionalAffineCoupling
import itertools

class PriorConditioner(nn.Module):
    """
    This PyTorch Module implements the neural network used to condition the
    base distribution of a NF as in Eq.(13).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PriorConditioner, self).__init__()

        #initialize linear transformations
        self.lin_input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)

        #initialize non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Given input x=[z_{t-1}, h_t], this method outputs mean and 
        std.dev of the diagonal gaussian base distribution of a NF.
        """
        hidden = self.relu(self.lin_input_to_hidden(x))
        hidden = self.relu(self.lin_hidden_to_hidden(hidden))
        loc = self.lin_hidden_to_loc(hidden)
        scale = self.softplus(self.lin_hidden_to_scale(hidden))
        return loc, scale

class ConditionalNormalizingFlow(nn.Module):
    """
    This PyTorch Module implements the NF defined by the triplet 
    [conditional affine coupling layer, batch norm, permute] as in Eq.(14).
    """
    def __init__(self, input_dim=2, split_dim=1, context_dim=1, hidden_dim=128, num_layers=1, flow_length=10, 
            use_cuda=False):
        super(ConditionalNormalizingFlow, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim)) # base distribution is Isotropic Gaussian
        self.prior_conditioner = PriorConditioner(context_dim, hidden_dim, input_dim)
        self.param_dims = [input_dim-split_dim, input_dim-split_dim]
        # Define series of bijective transformations
        self.transforms = [ConditionalAffineCoupling(split_dim, ConditionalDenseNN(split_dim, context_dim, [hidden_dim]*num_layers, self.param_dims)) for _ in range(flow_length)]
        self.perms = [permute(2, torch.tensor([1,0])) for _ in range(flow_length)]
        self.bns = [BatchNorm(input_dim=1) for _ in range(flow_length)]
        # Concatenate AffineCoupling layers with Permute and BatchNorm Layers
        self.generative_flows = list(itertools.chain(*zip(self.transforms, self.bns, self.perms)))[:-2] # generative direction (z-->x)
        self.normalizing_flows = self.generative_flows[::-1] # normalizing direction (x-->z)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
            nn.ModuleList(self.transforms).cuda()
    
    def model(self, X=None, H=None):
        N = len(X) if X is not None else None
        pyro.module("nf", nn.ModuleList(self.transforms))
        pyro.module("prior_conditioner", self.prior_conditioner)
        with pyro.plate("data", N):
                self.cond_flow_dist = self._condition(H)
                obs = pyro.sample("obs", self.cond_flow_dist, obs=X)
            
    def guide(self, X=None, H=None):
        pass
    
    def forward(self, z, H):
        zs = [z]
        _ = self._condition(H)
        for flow in self.generative_flows:
            z_i = flow(zs[-1])
            zs.append(z_i)
        return zs, z_i
    
    def backward(self, x, H):
        zs = [x]
        _ = self._condition(H)
        for flow in self.normalizing_flows:
            z_i = flow._inverse(zs[-1])
            zs.append(z_i)
        return zs, z_i
    
    def sample(self, num_samples, H):
        base_dist_loc, base_dist_scale = self.prior_conditioner(H)
        self.base_dist = dist.Normal(base_dist_loc, base_dist_scale)
        z_0_samples = self.base_dist.sample()
        zs, x = self.forward(z_0_samples, H)
        return x

    def log_prob(self, x, H):
        cond_flow_dist = self._condition(H)
        return cond_flow_dist.log_prob(x)
    
    def _condition(self, H):
        base_dist_loc, base_dist_scale = self.prior_conditioner(H)
        self.base_dist = dist.Normal(base_dist_loc, base_dist_scale)
        self.cond_transforms = [t.condition(H) for t in self.transforms]
        self.generative_flows = list(itertools.chain(*zip(self.cond_transforms, self.perms)))[:-1]
        self.normalizing_flows = self.generative_flows[::-1]
        return dist.TransformedDistribution(self.base_dist, self.generative_flows)

class Prior(nn.Module):
    """
    This PyTorch Module implements the conditional prior distribution
    over stochastic hidden states in the RFN generative model as in Eq.(12).
    """
    def __init__(self, lstm_dim, z_dim, hidden_dim=128):
        super(Prior, self).__init__()
        # initialize linear transformations
        self.lin_input_to_hidden = nn.Linear(lstm_dim + z_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, z_dim)
        # initialize non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
    
    def forward(self, z_t_1, h_t):
        """
        Given the previous stochastic state z_{t-1} and current deterministic state
        h_{t}, this method outputs mean and std.dev of a diagonal Gaussian over the
        stochastic state z_{t}.
        """
        input_ = torch.cat((z_t_1, h_t), dim=1)
        hidden = self.relu(self.lin_input_to_hidden(input_))
        loc = self.lin_hidden_to_loc(hidden)
        scale = self.softplus(self.lin_hidden_to_scale(hidden))
        return loc, scale

class FeatureExtractor(nn.Module):
    """
    This PyTorch Module implements the feature extractor as in Eq.(11).
    """
    def __init__(self, k):
        super(FeatureExtractor, self).__init__()
        self.lin_input_to_hidden = nn.Linear(k*k, 128)
        self.lin_hidden_to_hidden = nn.Linear(128, 128)
        self.lin_hidden_to_output = nn.Linear(128, 32)
        
        self.relu = nn.ReLU()
        self.k = k
        
    def forward(self, u):
        """Given the k x k spatial representation u, this method outputs a 32-dimensional
        embedding of extracted features.
        """
        u = u.view(-1, self.k*self.k)
        h = self.relu(self.lin_input_to_hidden(u))
        h = self.relu(self.lin_hidden_to_hidden(h))
        u = self.lin_hidden_to_output(h)
        return u

class Encoder(nn.Module):
    """
    This PyTorch Module implements the encoder network as in Eq.(15).
    """
    def __init__(self, z_dim, lstm_dim, hidden_dim=128):
        super(Encoder, self).__init__()
        # initialize linear transformations
        self.lin_input_to_hidden = nn.Linear(z_dim + lstm_dim + 32, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, z_dim)
        # initialize non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        
    def forward(self, z_t_1, h_t, x_t):
        """Given the previous stochastic state z_{t-1}, the current deterministic state h_{t}
        and current observation x_{t}, this method outputs mean and std.dev defining the approximate
        distribution over z_{t}.
        """
        input_ = torch.cat((z_t_1, h_t, x_t), dim=1)
        hidden = self.relu(self.lin_input_to_hidden(input_))
        loc = self.lin_hidden_to_loc(hidden)
        scale = self.softplus(self.lin_hidden_to_scale(hidden))
        return loc, scale

class RFN(nn.Module):
    """
    This PyTorch Module encapsulates the generative model as well as the
    variational distribution (the guide) for the Recurrent Flow Network.
    """
    def __init__(self, input_dim=2, z_dim=128, u_dim=32, prior_dim=128,
                decoder_dim=64, split_dim=1, decoder_layers=2, flow_length=35,
                encoder_dim=128, lstm_dim=128, k=64,
                use_cuda=True, annealing_factor=1.):
        super(RFN, self).__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.prior = Prior(lstm_dim, z_dim, prior_dim)
        self.decoder = ConditionalNormalizingFlow(input_dim, split_dim, z_dim+lstm_dim, 
                                                  decoder_dim, decoder_layers, flow_length)
        self.encoder = Encoder(z_dim, lstm_dim, encoder_dim)
        self.extractor = FeatureExtractor(k)
        self.p_lstm = nn.LSTM(u_dim, lstm_dim, batch_first=True, 
                            num_layers=1, bidirectional=False)
    
        # define (trainable) parameters z_0, z_q_0, h_0 and c_0 that help define the probability
        # distributions p(z_1) and q(z_1) and the initial hidden state for the LSTM
        # (since for t = 1 there are no previous latents to condition on)
        self.h_0 = nn.Parameter(torch.zeros(lstm_dim))
        self.c_0 = nn.Parameter(torch.zeros(lstm_dim))
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        
        # book-keeping
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.input_dim = input_dim
        self.k = k
        self.lstm_dim = lstm_dim
        self.use_cuda = use_cuda
        self.annealing_factor = annealing_factor
        # if on gpu cuda-ize all PyTorch (sub)modules
        if self.use_cuda:
            self.cuda()
            nn.ModuleList(self.decoder.transforms).cuda()
            nn.ModuleList(self.decoder.bns).cuda()
            self.decoder.base_dist = dist.Normal(torch.zeros(input_dim).cuda(),
                                         torch.ones(input_dim).cuda())
    
    def model(self, X=None, U=None, mask=None, batch_size=1):
        # get number of sequences
        N = len(U)
        # this is the number of time steps we need to process in the mini-batch
        T_max = U.size(1)
        # get input dimension 
        D = self.input_dim
        # get batch_size
        b = min(N, batch_size)
        
        assert U.shape == (N, T_max, self.k, self.k)
        
        # apply FeatureExtractor to 2d spatial representation of the data
        X_extr = torch.zeros((N, T_max, 32), device=U.device)
        for n in range(N):
            X_extr[n] = self.extractor(U[n]).view(1, T_max, 32)
            
        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("RFN", self)
        pyro.module("nf_transforms", nn.ModuleList(self.decoder.transforms))
        pyro.module("nf_batchnorms", nn.ModuleList(self.decoder.bns))
        
        h_prev = self.h_0.expand(b, self.h_0.size(0)).view(1, b, -1).contiguous()    
        c_prev = self.c_0.expand(b, self.c_0.size(0)).view(1, b, -1).contiguous()
        z_prev = self.z_0.expand(b, self.z_0.size(0)).contiguous()
        hidden_prev = (h_prev, c_prev)
        
        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        x_samples = torch.zeros((b, T_max, max(mask), D))
        with pyro.plate("data", N, dim=-2):
            # pyro.markov indicates conditional independence given by Markov property
            for t in pyro.markov(range(2, T_max+1)):
                p_lstm_input = X_extr[:, t-2].view(b, 1, self.u_dim)
                _, hidden_prev = self.p_lstm(p_lstm_input, hidden_prev)
                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1}, h_{t})
                z_loc, z_scale = self.prior(z_prev, hidden_prev[0].view(b, self.lstm_dim))
                # sample z_t ~ N(z_t | z_loc, z_scale) (scaled by KL annealing factor)
                with pyro.poutine.scale(scale=self.annealing_factor):
                    z_t = pyro.sample("z_%d"%t, dist.Normal(z_loc.view(b, 1, self.z_dim), z_scale.view(b, 1, self.z_dim)).to_event(1))
                    assert z_t.shape == (b, 1, self.z_dim)
                # observed data plate, mask encodes variable-length observations (i.e. different
                # number of lon-lat geo-coordinates for every time-step)
                with pyro.plate("density_%d"%t, size=mask[t-1], dim=-1):
                    flow_input = torch.cat((z_t.view(b, self.z_dim), hidden_prev[0].view(b, self.lstm_dim)), dim=1)
                    # conditon NF on stochastic and deterministic hidden states
                    self.decoder.cond_flow_dist = self.decoder._condition(flow_input)
                    x_dist = self.decoder.cond_flow_dist
                    # differentiate between "model sampling" and learning
                    if X is None:
                        x_samples[:, t-1, :mask[t-1]] = pyro.sample("x_%d"%(t), fn=x_dist, obs=None)
                    else:
                        x_samples[:, t-1, :mask[t-1]] = pyro.sample("x_%d"%(t), fn=x_dist, obs=X[:, t-1, :mask[t-1], :])
                    assert x_samples[:, t-1].shape == (b, max(mask), D)
                z_prev = z_t.view(b, -1) 
        return x_samples
    
    def guide(self, X=None, U=None, mask=None, batch_size=1):
        # get number of sequences
        N = len(U)
        # this is the number of time steps we need to process in the mini-batch
        T_max = U.size(1)
        # get input dimension
        D = self.input_dim
        # get batch_size
        b = min(N, batch_size)
        
        assert U.shape == (N, T_max, self.k, self.k)
        
        # apply FeatureExtractor to 2d spatial representation of the data
        X_extr = torch.zeros((N, T_max, 32), device=U.device)
        for n in range(N):
            X_extr[n] = self.extractor(U[n]).view(1, T_max, 32)
            
        # register all PyTorch (sub)modules with pyro
        pyro.module("RFN", self)
        pyro.module("nf_transforms", nn.ModuleList(self.decoder.transforms))
        pyro.module("nf_batchnorms", nn.ModuleList(self.decoder.bns))
        
        h_prev = self.h_0.expand(b, self.h_0.size(0)).view(1, b, -1).contiguous()    
        c_prev = self.c_0.expand(b, self.c_0.size(0)).view(1, b, -1).contiguous()
        hidden_prev = (h_prev, c_prev)
        z_prev = self.z_q_0.expand(b, self.z_q_0.size(0))
                
        Z_dists = []
        # the variational approximation follows the model's structure
        with pyro.plate("data", N, dim=-2):
            # pyro.markov indicates conditional independence given by Markov property
            for t in pyro.markov(range(2, T_max+1)):
                p_lstm_input = X_extr[:, t-2].view(b, 1, self.u_dim)
                _, hidden_prev = self.p_lstm(p_lstm_input, hidden_prev)
                # the next two lines assemble the distribution q(z_t | z_{t-1}, h_t, x_{t})
                z_loc, z_scale = self.encoder(z_prev, hidden_prev[0].view(b, self.lstm_dim), X_extr[:, t-1].view(b, self.u_dim))
                z_dist = dist.Normal(z_loc.view(b, 1, self.z_dim), z_scale.view(b, 1, self.z_dim)).to_event(1)
                Z_dists.append(z_dist)
                assert z_dist.event_shape == (self.z_dim,)
                assert z_dist.batch_shape == (b, 1)
                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=self.annealing_factor):
                    z_t = pyro.sample("z_%d" % t, z_dist)
                    if verbose: print("z_t: ", z_t.shape)
                z_prev = z_t.view(b, self.z_dim)
        return Z_dists
    
    def _get_log_likelihood(self, X, U, mask=None, num_particles=1):
        """
        Returns estimate of the marginal log-likelihood based on importance sampling.
        """
        trace_elbo = Trace_ELBO_Wrapper(num_particles=num_particles)
        mask = (torch.ones(X.shape[1], dtype=torch.int16))*(X.shape[2]) if mask is None else mask
        grid_log_probs = torch.zeros((X.shape[1], X.shape[2]))
        for model_trace, _ in trace_elbo._get_traces(self.model, self.guide, [X, U, mask], {}):
            for i in range(X.shape[1]-1):
                grid_log_probs[i, :mask[i+1]] = model_trace.nodes["x_%d"%(i+2)]["log_prob"]
        return grid_log_probs
