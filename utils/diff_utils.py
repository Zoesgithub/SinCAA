import numpy as np
import torch

class GaussianDDPM():
    def __init__(self, 
                 min_b:float,
                 max_b:float,
                 coordinate_scaling:float,
                 
                 ) -> None:
        self.min_b=min_b
        self.max_b=max_b
        self.coordinate_scaling=coordinate_scaling
    
    def sample_t(self, n:int, device):
        return torch.rand(n).to(device)

    def _scale(self, x):
        return x * self.coordinate_scaling

    def _unscale(self, x):
        return x / self.coordinate_scaling

    def b_t(self, t):
        return self.min_b + t*(self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t)[..., None] * x

    def sample_ref(self, n_samples: float, device):
        return torch.randn(size=(n_samples, 3)).to(device)

    def marginal_b_t(self, t):
        return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)

    def calc_x_0(self, score_t, x_t, t):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None]
        exp_fn = torch.exp 
        cond_var = 1 - exp_fn(-beta_t)
        return self._unscale((score_t * cond_var + self._scale(x_t)) / exp_fn(-1/2*beta_t))

    def forward(self, x_t_1: torch.Tensor, t: torch.Tensor, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.randn(x_t_1.shape).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: torch.Tensor, t: torch.Tensor):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        x_0 = self._scale(x_0)
        x_t =  torch.distributions.Normal(
            loc=torch.exp(-1/2*self.marginal_b_t(t))[..., None] * x_0,
            scale=(torch.sqrt(1 - torch.exp(-self.marginal_b_t(t)))[..., None]).expand_as(x_0).clamp(1e-6)
        ).sample()
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        assert score_t.shape==x_t.shape, f"{score_t.shape} {x_t.shape}"
        return x_t, score_t

    def score_scaling(self, t: float):
        return 1 / np.sqrt(self.conditional_var(t))

    def reverse(
            self,
            x_t: torch.Tensor,
            score_t: torch.Tensor,
            t: torch.Tensor,
            dt: torch.Tensor,
            mask: torch.Tensor=None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.randn(size=score_t.shape).to(x_t.device)
        perturb = (f_t - g_t[..., None]**2 * score_t) * dt[..., None] + g_t[..., None]  * torch.sqrt(dt[..., None] ) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = x_t.new_ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = torch.sum(x_t_1, axis=-2) / torch.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        
        return 1 - torch.exp(-self.marginal_b_t(t))
        

    def score(self, x_t, x_0, t, scale=False):
        
        exp_fn = torch.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1/2*self.marginal_b_t(t))[..., None] * x_0) / self.conditional_var(t)[..., None]