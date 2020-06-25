"""
Utlity Functions for Recurrent Flow Networks 
------------------------------------------------------

Requirements:
- torch==1.4.0
- pyro-ppl==1.3.1
- numpy==1.18.2
- pandas==1.0.3
- matplotlib==3.0.3

"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyro.util import warn_if_nan
from pyro.infer import Trace_ELBO

class RFNDataset(Dataset):
    """Spatio-temporal demand modelling dataset."""
    def __init__(self, X, U):
        """
        """
        self.X = X
        self.U = U

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X_i, U_i = self.X[idx].float(), self.U[idx].float()
        return X_i, U_i
    
class Trace_ELBO_Wrapper(Trace_ELBO):
    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the (Negative) ELBO, KL divergence and Marginal Log-Likelihood.
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        log_prob_sum = 0.0
        kl_sum = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            log_prob = model_trace.log_prob_sum()
            log_prob_sum += log_prob
            kl = guide_trace.log_prob_sum()
            kl_sum += kl
            elbo_particle = log_prob - kl
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss, kl_sum, log_prob_sum
    
    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the (Negative) ELBO, KL divergence and Marginal Log-Likelihood.
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        log_prob_sum = 0.0
        kl_sum = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            log_prob = model_trace.log_prob_sum()
            log_prob_sum += log_prob / self.num_particles
            kl = guide_trace.log_prob_sum()
            kl_sum += kl / self.num_particles
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(model_trace, guide_trace)
            loss += loss_particle / self.num_particles
            
            # collect parameters to train from model and guide
            trainable_params = any(site["type"] == "param"
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values())

            if trainable_params and getattr(surrogate_loss_particle, 'requires_grad', False):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
                
        warn_if_nan(loss, "loss")
        return loss, kl_sum, log_prob_sum

def plot_heatmap(grid, pts, mask, log_probs, t=0):
    df_grid = pd.concat((pd.DataFrame(grid[0, 0, :, 0].detach().numpy(), columns=["lat"]),
                         pd.DataFrame(grid[0, 0, :, 1].detach().numpy(), columns=["lon"]),
                         pd.DataFrame(log_probs[t].detach().numpy(), columns=["log_prob"])), axis=1)
    df_grid.head()

    plt.figure(figsize=(10, 10))
    plt.scatter(-pts[0, t+1, :mask[t+1], 1], pts[0, t+1, :mask[t+1], 0], c="r", s=10)
    im = plt.imshow(np.flip(df_grid["log_prob"].values.reshape(110, 110), axis=1), origin="lower", interpolation="bicubic",
                    alpha=1, vmax=10., vmin=-10., cmap="seismic", extent=(-1, 0, 0, 1))
    plt.xticks([])
    plt.yticks([])
    plt.title("Spatio-Temporal Density Estimation (t={}:00)".format(2*t%24))
    plt.show()
    