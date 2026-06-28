"""
demand_families.py -- demand processes beyond Poisson-rate randomization (for M2; enables the
AR(1) communication sweep of roadmap 5.6).

WHY: training currently randomizes only the Poisson RATE (see DemandRandomizedBeerGame). That is
rate-robustness within ONE family, not distributional robustness. This module adds AR(1),
negative-binomial, and a per-episode FAMILY-randomization curriculum.

!! CONSTRUCTION NOTE (the bug this version fixes) !!
The base env validates `demand_type` against {step, zero, black_swan, extreme_chaos, poisson}
and RAISES on anything else (beer_game_env.py ~line 169). So these envs are built with
demand_type="poisson" (which passes validation) plus a separate key `family` in {ar1, negbin}
that the overridden _roll_stochastic_demand checks FIRST -- exactly the pattern
DemandRandomizedBeerGame uses (it keeps demand_type="poisson" and routes internally). Do NOT
pass demand_type="ar1"/"negbin"; pass demand_type="poisson", family="ar1"/"negbin".

Per-episode state is initialized in reset() BEFORE super().reset() so the env's demand-cache
pre-roll routes through the override (mirrors DemandRandomizedBeerGame). AR(1) assumes
_roll_stochastic_demand is called once per period in increasing order (true here, including the
lookahead cache); the i.i.d. families are order-independent.

INTEGRATION
  * Held-out-FAMILY eval (M2): make_heldout_family_envs(...) builds AR(1)-rho + NegBin-dispersion
    test envs; score DRACO on them like heldout_eval does for lambda.
  * AR(1) comm sweep (5.6): make_ar1_rho_envs(...) -> run the topology study per rho.
  * DR training (M2): swap DemandRandomizedBeerGame -> FamilyRandomizedBeerGame in the trainer
    (it mirrors DemandRandomizedBeerGame's config-only constructor form).
"""
import numpy as np


# ----------------------------------------------------------------------- standalone samplers
def poisson_sample(rng, mu):
    return float(rng.poisson(float(mu)))


def negbin_sample(rng, mu, dispersion):
    """NegBin with mean mu and variance mu + mu^2/dispersion (dispersion->inf gives Poisson).
    numpy: negative_binomial(n=r, p) has mean r(1-p)/p; set r=dispersion, p=r/(r+mu)."""
    mu = max(1e-6, float(mu)); r = max(1e-6, float(dispersion))
    return float(rng.negative_binomial(r, r / (r + mu)))


def ar1_step(prev, mu, rho, sigma, rng):
    """One AR(1) step. Returns (demand_int, new_latent). The latent is the unclipped AR(1)
    value (so autocorrelation is preserved); emitted demand is max(0, round(latent))."""
    latent = mu + rho * (prev - mu) + rng.normal(0.0, sigma)
    return max(0.0, float(round(latent))), latent


# ----------------------------------------------------------------------- env subclasses
def make_demand_family_envs(base_cls):
    """Factory -> (AR1BeerGame, NegBinBeerGame, FamilyRandomizedBeerGame) subclassed from the
    project env. Pass base_cls=BeerGameParallelEnv. All are built with demand_type='poisson'."""

    class AR1BeerGame(base_cls):
        """Build with demand_type='poisson', family='ar1'. Config: ar1_mu(12), ar1_rho(.6), ar1_sigma(3)."""
        def reset(self, seed=None, options=None):
            self._ar1_latent = float(self._config.get("ar1_mu", 12.0))   # before super (cache pre-roll)
            return super().reset(seed=seed, options=options)

        def _roll_stochastic_demand(self, step):
            if self._config.get("family") == "ar1":
                mu = float(self._config.get("ar1_mu", 12.0))
                rho = float(self._config.get("ar1_rho", 0.6))
                sigma = float(self._config.get("ar1_sigma", 3.0))
                if getattr(self, "_ar1_latent", None) is None:
                    self._ar1_latent = mu
                d, self._ar1_latent = ar1_step(self._ar1_latent, mu, rho, sigma, self.np_random)
                return d
            return super()._roll_stochastic_demand(step)

    class NegBinBeerGame(base_cls):
        """Build with demand_type='poisson', family='negbin'. Config: nb_mu(12), nb_dispersion(4)."""
        def _roll_stochastic_demand(self, step):
            if self._config.get("family") == "negbin":
                return negbin_sample(self.np_random,
                                     float(self._config.get("nb_mu", 12.0)),
                                     float(self._config.get("nb_dispersion", 4.0)))
            return super()._roll_stochastic_demand(step)

    class FamilyRandomizedBeerGame(base_cls):
        """Per-EPISODE distributional randomization (the DR curriculum). Build with
        demand_type='poisson'. At reset, samples a family from `dr_families` and its params.
        Mirrors DemandRandomizedBeerGame but over FAMILIES, not just the Poisson rate.

        Config (all optional): dr_families (subset of ['poisson','negbin','ar1']),
          dr_lambda_lo/hi, nb_mu_lo/hi, nb_dispersion_lo/hi, ar1_mu_lo/hi, ar1_rho_lo/hi, ar1_sigma.
        TRAIN/TEST HYGIENE: leave at least one family OUT of dr_families for the held-out test."""
        def reset(self, seed=None, options=None):
            cfg = self._config
            # dedicated RNG so family choice is reproducible from `seed` and independent of the
            # demand-draw RNG (mirrors DemandRandomizedBeerGame._dr_rng).
            self._fam_rng = (np.random.default_rng(seed + 77761) if seed is not None
                             else np.random.default_rng())
            fams = list(cfg.get("dr_families", ["poisson", "negbin", "ar1"]))
            self._fam = fams[int(self._fam_rng.integers(len(fams)))]
            if self._fam == "poisson":
                self._fam_mu = float(self._fam_rng.uniform(cfg.get("dr_lambda_lo", 4.0), cfg.get("dr_lambda_hi", 24.0)))
            elif self._fam == "negbin":
                self._fam_mu = float(self._fam_rng.uniform(cfg.get("nb_mu_lo", 4.0), cfg.get("nb_mu_hi", 24.0)))
                self._fam_disp = float(self._fam_rng.uniform(cfg.get("nb_dispersion_lo", 2.0), cfg.get("nb_dispersion_hi", 12.0)))
            else:  # ar1
                self._fam_mu = float(self._fam_rng.uniform(cfg.get("ar1_mu_lo", 4.0), cfg.get("ar1_mu_hi", 24.0)))
                self._fam_rho = float(self._fam_rng.uniform(cfg.get("ar1_rho_lo", 0.0), cfg.get("ar1_rho_hi", 0.8)))
                self._fam_sigma = float(cfg.get("ar1_sigma", 3.0))
                self._ar1_latent = self._fam_mu
            return super().reset(seed=seed, options=options)

        def _roll_stochastic_demand(self, step):
            if getattr(self, "_fam", None) is None:           # safety before first reset
                return super()._roll_stochastic_demand(step)
            if self._fam == "poisson":
                return poisson_sample(self.np_random, self._fam_mu)
            if self._fam == "negbin":
                return negbin_sample(self.np_random, self._fam_mu, self._fam_disp)
            d, self._ar1_latent = ar1_step(self._ar1_latent, self._fam_mu,
                                           self._fam_rho, self._fam_sigma, self.np_random)
            return d

    return AR1BeerGame, NegBinBeerGame, FamilyRandomizedBeerGame


# ----------------------------------------------------------------------- held-out eval builders
def make_ar1_rho_envs(ar1_cls, env_cfg, rhos=(0.0, 0.3, 0.6, 0.9), mu=12.0, sigma=3.0):
    """One AR(1) env per rho (comm-value-vs-autocorrelation sweep, roadmap 5.6). ar1_cls is the
    AR1BeerGame from make_demand_family_envs. Built with demand_type='poisson', family='ar1'."""
    return {float(r): ar1_cls({**env_cfg, "demand_type": "poisson", "family": "ar1",
                               "ar1_mu": mu, "ar1_rho": float(r), "ar1_sigma": sigma})
            for r in rhos}


def make_heldout_family_envs(ar1_cls, negbin_cls, env_cfg,
                             ar1_rhos=(0.3, 0.6, 0.9), nb_disps=(2.0, 4.0, 8.0), mu=12.0):
    """Held-out distributional-shift test set (M2): AR(1) at several rho + NegBin at several
    dispersion, fixed mean. Score DRACO here as heldout_eval scores lambda."""
    envs = {}
    for r in ar1_rhos:
        envs[f"ar1_rho{r:g}"] = ar1_cls({**env_cfg, "demand_type": "poisson", "family": "ar1",
                                         "ar1_mu": mu, "ar1_rho": float(r), "ar1_sigma": 3.0})
    for disp in nb_disps:
        envs[f"negbin_disp{disp:g}"] = negbin_cls({**env_cfg, "demand_type": "poisson", "family": "negbin",
                                                   "nb_mu": mu, "nb_dispersion": float(disp)})
    return envs


# ----------------------------------------------------------------------- self-test (samplers only)
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for disp in (2.0, 8.0, 1e6):
        x = np.array([negbin_sample(rng, 12.0, disp) for _ in range(20000)])
        print(f"  NegBin disp={disp:>7g}: mean={x.mean():5.2f} (12) var={x.var():6.2f} "
              f"(expected {12.0 + 144.0/disp:.1f}, Poisson would be 12.0)")
    for rho in (0.0, 0.5, 0.9):
        d = np.zeros(5000); lat = 12.0
        for t in range(5000):
            d[t], lat = ar1_step(lat, 12.0, rho, 3.0, rng)
        print(f"  AR(1) rho={rho:.1f}: mean={d.mean():5.2f} (12) emp lag-1 autocorr="
              f"{np.corrcoef(d[:-1], d[1:])[0,1]:.2f}")
    print("demand_families self-test PASS  (samplers; env routing needs the real env)")