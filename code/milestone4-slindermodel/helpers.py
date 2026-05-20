# Boilerplate code
from __future__ import annotations
from typing import Sequence, Tuple

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# data helpers

from math import radians, cos, degrees
from typing import Iterable, Tuple, List

NorthEast = Tuple[float, float]  # (north_m, east_m)
LonLat = Tuple[float, float]     # (longitude, latitude)

LonLat = Tuple[float, float]      # (longitude, latitude)
NorthEast = Tuple[float, float]   # (north_m, east_m)


def lonlat_to_north_east(
    points: Iterable[LonLat],
    reference: LonLat,
    earth_radius_m: float = 6_378_137.0,
) -> List[NorthEast]:
    """
    Convert longitude/latitude points to meters north/east of a reference point.

    Parameters
    ----------
    points:
        Iterable of (longitude, latitude) points in degrees.
    reference:
        Reference point as (longitude, latitude) in degrees.
        This is treated as the local origin.
    earth_radius_m:
        Earth radius in meters. Defaults to WGS84 equatorial radius.

    Returns
    -------
    List of (north_m, east_m) tuples.
    """

    ref_lon, ref_lat = reference

    ref_lat_rad = radians(ref_lat)
    meters_per_rad_lat = earth_radius_m
    meters_per_rad_lon = earth_radius_m * cos(ref_lat_rad)

    result = []

    for lon, lat in points:
        d_lat_rad = radians(lat - ref_lat)
        d_lon_rad = radians(lon - ref_lon)

        north_m = d_lat_rad * meters_per_rad_lat
        east_m = d_lon_rad * meters_per_rad_lon

        result.append((north_m, east_m))

    return result


def north_east_to_lonlat(
    points_m: Iterable[NorthEast],
    reference: LonLat,
    earth_radius_m: float = 6_378_137.0,
) -> List[LonLat]:
    """
    Convert local meter offsets back to longitude/latitude.

    Parameters
    ----------
    points_m:
        Iterable of (north_m, east_m) points.
    reference:
        Reference point as (longitude, latitude) in degrees.
    earth_radius_m:
        Earth radius in meters.

    Returns
    -------
    List of (longitude, latitude) tuples.
    """

    ref_lon, ref_lat = reference

    ref_lat_rad = radians(ref_lat)
    meters_per_rad_lat = earth_radius_m
    meters_per_rad_lon = earth_radius_m * cos(ref_lat_rad)

    result = []

    for north_m, east_m in points_m:
        d_lat_rad = north_m / meters_per_rad_lat
        d_lon_rad = east_m / meters_per_rad_lon

        lat = ref_lat + degrees(d_lat_rad)
        lon = ref_lon + degrees(d_lon_rad)

        result.append((lon, lat))

    return result

# -----------------------------
# Data containers
# -----------------------------


@dataclass
class APObs:
    """
    Observations for one access point.

    Required for both estimators:
        ap_id: human-readable AP identifier
        xy: tensor [n, 2], local meter coordinates
        y: tensor [n], RSSI values in dBm
        outdoor: tensor [n], 1 if measurement is outdoor, 0 if indoor

    Required only for HierarchicalGridMAPEstimator:
        grid_xy: tensor [J, 2], candidate indoor AP locations
        d2_grid: tensor [n, J], squared distances ||xy_i - grid_j||^2
        logpi_grid: tensor [J], log prior probabilities over grid candidates
    """
    ap_id: str
    xy: torch.Tensor
    y: torch.Tensor
    outdoor: torch.Tensor

    grid_xy: Optional[torch.Tensor] = None
    d2_grid: Optional[torch.Tensor] = None
    logpi_grid: Optional[torch.Tensor] = None


# -----------------------------
# Utilities
# -----------------------------

def choose_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device_ap_data(
    ap_data: Sequence[APObs],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> List[APObs]:
    out = []
    for ap in ap_data:
        out.append(
            APObs(
                ap_id=ap.ap_id,
                xy=ap.xy.to(device=device, dtype=dtype),
                y=ap.y.to(device=device, dtype=dtype),
                outdoor=ap.outdoor.to(device=device, dtype=dtype),
                grid_xy=None if ap.grid_xy is None else ap.grid_xy.to(
                    device=device, dtype=dtype),
                d2_grid=None if ap.d2_grid is None else ap.d2_grid.to(
                    device=device, dtype=dtype),
                logpi_grid=None if ap.logpi_grid is None else ap.logpi_grid.to(
                    device=device, dtype=dtype),
            )
        )
    return out


def positive(raw: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    return F.softplus(raw) + min_value


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate inverse of softplus for positive x.
    """
    return torch.log(torch.expm1(x.clamp_min(1e-8)))


def normal_nlp(x: torch.Tensor, mean: float | torch.Tensor, sd: float | torch.Tensor) -> torch.Tensor:
    """
    Negative log density up to constants for Normal(mean, sd^2).
    Includes log(sd), which matters when sd is learned.
    """
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    sd = torch.as_tensor(sd, device=x.device, dtype=x.dtype)
    return 0.5 * ((x - mean) / sd).pow(2) + torch.log(sd)


def halfnormal_nlp(x: torch.Tensor, sd: float | torch.Tensor) -> torch.Tensor:
    """
    Negative log density up to constants for HalfNormal(sd), assuming x >= 0.
    """
    sd = torch.as_tensor(sd, device=x.device, dtype=x.dtype)
    return 0.5 * (x / sd).pow(2) + torch.log(sd)


def _finite_ap_xy_y(xy: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keeps only observations with finite xy and finite RSSI.
    """
    mask = torch.isfinite(xy).all(dim=1) & torch.isfinite(y)
    return xy[mask], y[mask]


def _aggregate_by_rounded_location(
    xy: torch.Tensor,
    y: torch.Tensor,
    tol_m: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregates repeated / near-repeated locations.

    Returns:
        unique_xy: [M, 2]
            Representative rounded unique locations.
        unique_y_mean: [M]
            Mean RSSI at each unique location.
        counts: [M]
            Number of raw observations at each unique location.

    Note:
        This intentionally runs the grouping on CPU for robustness.
        It is only used during initialization/filtering, so speed is not important.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape [n, 2], got {tuple(xy.shape)}")

    if y.ndim != 1 or y.shape[0] != xy.shape[0]:
        raise ValueError(
            f"y must have shape [n], got {tuple(y.shape)} for xy shape {tuple(xy.shape)}")

    device = xy.device
    dtype = xy.dtype

    xy_cpu = xy.detach().cpu()
    y_cpu = y.detach().cpu()

    if xy_cpu.shape[0] == 0:
        return (
            torch.empty((0, 2), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=dtype),
        )

    rounded = torch.round(xy_cpu / tol_m) * tol_m

    unique_xy_cpu, inverse = torch.unique(
        rounded,
        dim=0,
        return_inverse=True,
    )

    M = unique_xy_cpu.shape[0]

    y_sum = torch.zeros(M, dtype=y_cpu.dtype)
    counts = torch.zeros(M, dtype=y_cpu.dtype)

    y_sum.index_add_(0, inverse, y_cpu)
    counts.index_add_(0, inverse, torch.ones_like(y_cpu))

    unique_y_mean_cpu = y_sum / counts.clamp_min(1.0)

    return (
        unique_xy_cpu.to(device=device, dtype=dtype),
        unique_y_mean_cpu.to(device=device, dtype=dtype),
        counts.to(device=device, dtype=dtype),
    )


def num_unique_locations(
    xy: torch.Tensor,
    y: torch.Tensor | None = None,
    tol_m: float = 1.0,
) -> int:
    """
    Counts approximately unique geographic locations.

    If y is provided, drops non-finite xy/y rows.
    If y is None, only drops non-finite xy rows.
    """
    if y is not None:
        xy, y = _finite_ap_xy_y(xy, y)
    else:
        xy = xy[torch.isfinite(xy).all(dim=1)]

    if xy.shape[0] == 0:
        return 0

    xy_cpu = xy.detach().cpu()
    rounded = torch.round(xy_cpu / tol_m) * tol_m
    unique_xy = torch.unique(rounded, dim=0)

    return int(unique_xy.shape[0])


def keep_for_dense_sse_baseline(
    ap: "APObs",
    min_unique_locations: int = 8,
    tol_m: float = 1.0,
) -> bool:
    M_a = num_unique_locations(ap.xy, ap.y, tol_m=tol_m)
    return M_a >= min_unique_locations


def initialize_b_w_ell(
    ap_data: Sequence["APObs"],
    tol_m: float = 1.0,
    default_ell_m: float = 15.0,
    min_ell_m: float = 5.0,
    max_ell_m: float = 100.0,
    min_w_db: float = 3.0,
    softmax_temperature_db: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Safe drop-in replacement for the old initializer.

    Returns:
        b0:       [A]
        logw0:    [A]
        logell0:  [A]
        mu0:      [A, 2]

    Model:
        f_a(x) = b_a + w_a * exp(-||x - mu_a||^2 / (2 ell_a^2) - gamma * outdoor)

    Notes:
        - Uses approximately unique geographic locations, not raw observations.
        - Does not estimate a covariance matrix.
        - Uses a default ell if spatial spread is too small.
        - Avoids NaNs from repeated locations or degenerate geography.
    """
    b0s = []
    logw0s = []
    logell0s = []
    mu0s = []

    for ap in ap_data:
        xy, y = _finite_ap_xy_y(ap.xy, ap.y)

        if xy.shape[0] == 0:
            raise ValueError(f"AP {ap.ap_id!r} has zero finite observations.")

        unique_xy, unique_y, counts = _aggregate_by_rounded_location(
            xy,
            y,
            tol_m=tol_m,
        )

        M = unique_xy.shape[0]

        if M == 0:
            raise ValueError(
                f"AP {ap.ap_id!r} has zero unique finite locations.")

        # Baseline RSSI: low quantile of unique-location mean RSSI.
        # This is safer than using raw observations because async joins can create many duplicates.
        b0 = torch.quantile(unique_y, 0.10)

        # Peak-ish RSSI.
        peak0 = torch.quantile(unique_y, 0.90)

        # Positive bump size above baseline.
        w0 = (peak0 - b0).clamp_min(min_w_db).clamp_max(40)

        # RSSI-weighted centroid.
        # Stronger RSSI is larger / less negative, so softmax emphasizes stronger points.
        if M == 1:
            mu0 = unique_xy[0]
        else:
            centered_y = unique_y - unique_y.mean()
            weights = torch.softmax(centered_y / softmax_temperature_db, dim=0)

            if not torch.isfinite(weights).all() or weights.sum() <= 0:
                weights = torch.ones_like(unique_y) / M

            mu0 = (weights[:, None] * unique_xy).sum(dim=0)

            if not torch.isfinite(mu0).all():
                mu0 = unique_xy.mean(dim=0)

        # Spatial length-scale initialization.
        # Use the RMS distance of unique locations around the initialized mu.
        if M < 2:
            ell0 = torch.tensor(
                default_ell_m, device=xy.device, dtype=xy.dtype)
        else:
            d2 = (unique_xy - mu0[None, :]).pow(2).sum(dim=1)
            rms_dist = torch.sqrt(d2.mean().clamp_min(0.0))

            if not torch.isfinite(rms_dist) or rms_dist <= 0:
                ell0 = torch.tensor(
                    default_ell_m, device=xy.device, dtype=xy.dtype)
            else:
                ell0 = rms_dist.clamp(min_ell_m, max_ell_m)

        if not torch.isfinite(b0):
            b0 = torch.nanmedian(y)

        if not torch.isfinite(w0) or w0 <= 0:
            w0 = torch.tensor(min_w_db, device=xy.device, dtype=xy.dtype)

        if not torch.isfinite(ell0) or ell0 <= 0:
            ell0 = torch.tensor(
                default_ell_m, device=xy.device, dtype=xy.dtype)

        if not torch.isfinite(mu0).all():
            mu0 = xy.mean(dim=0)

        b0s.append(b0)
        logw0s.append(torch.log(w0))
        logell0s.append(torch.log(ell0))
        mu0s.append(mu0)

    return (
        torch.stack(b0s),
        torch.stack(logw0s),
        torch.stack(logell0s),
        torch.stack(mu0s),
    )


def fit_torch_module(
    model: nn.Module,
    steps: int = 2000,
    lr: float = 1e-2,
    optimizer_name: str = "adam",
    print_every: int = 200,
) -> List[float]:
    """
    Generic optimizer loop.

    optimizer_name:
        "adam"  - good default for first pass
        "lbfgs" - can polish after Adam; uses PyTorch closure API
    """
    history = []

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss()
            loss.backward()
            optimizer.step()

            value = float(loss.detach().cpu())
            history.append(value)

            if print_every and step % print_every == 0:
                print(f"[Adam] step={step:05d} loss={value:.4f}")

    elif optimizer_name.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=steps,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        history.append(float(loss.detach().cpu()))
        print(f"[LBFGS] final loss={history[-1]:.4f}")

    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name!r}")

    return history


class DenseAPSSEEstimator(nn.Module):
    """
    Unregularized squared-error baseline.

    Fits only APs with at least min_obs observations.

    Objective:
        sum_{a,i} (y_ai - f_a(x_ai))^2

    Parameters per AP:
        b_a
        log_w_a
        log_ell_a
        mu_a in R^2

    Global by default:
        gamma >= 0, outdoor attenuation

    Optional per AP:
        gamma_a >= 0, AP-specific outdoor attenuation
    """

    def __init__(
        self,
        ap_data: Sequence[APObs],
        min_obs: int = 10,
        learn_gamma: bool = True,
        shared_gamma: bool = True,
        gamma_init: float | Sequence[float] | torch.Tensor = 0.5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.device = device or choose_device()
        self.dtype = dtype

        dense = [ap for ap in ap_data if len(ap.y) >= min_obs]
        if len(dense) == 0:
            raise ValueError(
                f"No APs have at least min_obs={min_obs} observations.")

        self.ap_data = to_device_ap_data(dense, self.device, self.dtype)
        self.ap_ids = [ap.ap_id for ap in self.ap_data]
        self.A = len(self.ap_data)
        self.learn_gamma = learn_gamma
        self.shared_gamma = shared_gamma

        b0, logw0, logell0, mu0 = initialize_b_w_ell(self.ap_data)

        self.b = nn.Parameter(b0.clone())
        self.log_w = nn.Parameter(logw0.clone())
        self.log_ell = nn.Parameter(logell0.clone())
        self.mu = nn.Parameter(mu0.clone())

        gamma0 = torch.as_tensor(
            gamma_init,
            device=self.device,
            dtype=self.dtype,
        )
        if self.shared_gamma:
            if gamma0.ndim != 0:
                raise ValueError(
                    "gamma_init must be scalar when shared_gamma=True.")
        else:
            if gamma0.ndim == 0:
                gamma0 = gamma0.repeat(self.A)
            elif gamma0.shape != (self.A,):
                raise ValueError(
                    "gamma_init must be scalar or have shape (n_aps,) "
                    "when shared_gamma=False."
                )

        self.raw_gamma = nn.Parameter(inverse_softplus(gamma0))

        if not learn_gamma:
            self.raw_gamma.requires_grad_(False)

    @property
    def gamma(self) -> torch.Tensor:
        return positive(self.raw_gamma)

    def gamma_for_ap(self, a: int) -> torch.Tensor:
        gamma = self.gamma
        if self.shared_gamma:
            return gamma
        return gamma[a]

    def ap_params(self) -> Dict[str, torch.Tensor]:
        return {
            "b": self.b,
            "w": torch.exp(self.log_w),
            "ell": torch.exp(self.log_ell),
            "mu": self.mu,
            "gamma": self.gamma,
        }

    def predict_points(self, a: int, xy, outdoor):
        b = self.b[a]
        w = torch.exp(self.log_w[a])
        ell = torch.exp(self.log_ell[a])
        mu = self.mu[a]
        gamma = self.gamma_for_ap(a)

        d2 = (xy - mu[None, :]).pow(2).sum(dim=1).clip(0,
                                                       ell.detach() ** 2 * 20)
        kernel = torch.exp(
            -0.5 * d2 / ell.pow(2)
            - gamma * outdoor
        )

        return b + w * kernel

    def predict_ap(self, a: int) -> torch.Tensor:
        ap = self.ap_data[a]

        b = self.b[a]
        w = torch.exp(self.log_w[a])
        ell = torch.exp(self.log_ell[a])
        mu = self.mu[a]
        gamma = self.gamma_for_ap(a)

        d2 = (ap.xy - mu[None, :]).pow(2).sum(dim=1).clip(0,
                                                          ell.detach() ** 2 * 20)

        kernel = torch.exp(
            -0.5 * d2 / ell.pow(2)
            - gamma * ap.outdoor
        )

        return b + w * kernel

    def loss(self) -> torch.Tensor:
        total = torch.zeros((), device=self.device, dtype=self.dtype)

        for a in range(self.A):
            ap = self.ap_data[a]
            pred = self.predict_ap(a)
            resid = ap.y - pred
            total = total + resid.pow(2).sum()

        return total

    @torch.no_grad()
    def summary(self) -> Dict[str, Any]:
        params = self.ap_params()

        b = params["b"].detach().cpu()
        w = params["w"].detach().cpu()
        ell = params["ell"].detach().cpu()
        mu = params["mu"].detach().cpu()
        gamma = params["gamma"].detach().cpu()

        out = {
            "shared_gamma": self.shared_gamma,
            "gamma": float(gamma) if self.shared_gamma else [float(g) for g in gamma],
            "aps": {}
        }

        for i, ap_id in enumerate(self.ap_ids):
            ap_gamma = gamma if self.shared_gamma else gamma[i]
            out["aps"][ap_id] = {
                "b": float(b[i]),
                "w": float(w[i]),
                "ell": float(ell[i]),
                "mu_x": float(mu[i, 0]),
                "mu_y": float(mu[i, 1]),
                "gamma": float(ap_gamma),
                "rho_outdoor_retention": float(torch.exp(-ap_gamma)),
                "n_obs": len(self.ap_data[i].y),
            }

        return out


class HierarchicalGridMAPEstimator(nn.Module):
    """
    Hierarchical MAP estimator with discrete candidate AP locations.

    For each AP a:
        z_a ~ Categorical(pi_a)
        mu_a = grid_xy[a][z_a]

        b_a      ~ Normal(m_b, s_b^2)
        log_w_a  ~ Normal(m_log_w, s_log_w^2)
        log_ell_a ~ Normal(m_log_ell, s_log_ell^2)

        y_ai ~ Normal(
            b_a + exp(log_w_a) * exp(
                -||x_ai - mu_a||^2 / (2 exp(log_ell_a)^2)
                - gamma * outdoor_ai
            ),
            tau^2
        )

    This is MAP, not posterior sampling.
    """

    def __init__(
        self,
        ap_data: Sequence[APObs],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        init_tau: float = 6.0,
        init_gamma: float = 0.5,
    ):
        super().__init__()

        self.device = device or choose_device()
        self.dtype = dtype

        # Need grid fields for this estimator.
        for ap in ap_data:
            if ap.d2_grid is None or ap.logpi_grid is None or ap.grid_xy is None:
                raise ValueError(
                    f"AP {ap.ap_id!r} is missing grid_xy, d2_grid, or logpi_grid."
                )

        self.ap_data = to_device_ap_data(ap_data, self.device, self.dtype)
        self.ap_ids = [ap.ap_id for ap in self.ap_data]
        self.A = len(self.ap_data)

        # Initialize AP-level parameters from observations.
        b0, logw0, logell0, _ = initialize_b_w_ell(self.ap_data)

        self.b = nn.Parameter(b0.clone())
        self.log_w = nn.Parameter(logw0.clone())
        self.log_ell = nn.Parameter(logell0.clone())

        # Hyperparameters.
        self.m_b = nn.Parameter(
            torch.tensor(-90.0, device=self.device, dtype=self.dtype))
        self.raw_s_b = nn.Parameter(inverse_softplus(
            torch.tensor(10.0, device=self.device, dtype=self.dtype)))

        self.m_log_w = nn.Parameter(torch.tensor(
            math.log(30.0), device=self.device, dtype=self.dtype))
        self.raw_s_log_w = nn.Parameter(inverse_softplus(
            torch.tensor(1.0, device=self.device, dtype=self.dtype)))

        self.m_log_ell = nn.Parameter(torch.tensor(
            math.log(15.0), device=self.device, dtype=self.dtype))
        self.raw_s_log_ell = nn.Parameter(inverse_softplus(
            torch.tensor(1.0, device=self.device, dtype=self.dtype)))

        self.raw_tau = nn.Parameter(inverse_softplus(
            torch.tensor(init_tau, device=self.device, dtype=self.dtype)))
        self.raw_gamma = nn.Parameter(inverse_softplus(
            torch.tensor(init_gamma, device=self.device, dtype=self.dtype)))

    @property
    def s_b(self) -> torch.Tensor:
        return positive(self.raw_s_b)

    @property
    def s_log_w(self) -> torch.Tensor:
        return positive(self.raw_s_log_w)

    @property
    def s_log_ell(self) -> torch.Tensor:
        return positive(self.raw_s_log_ell)

    @property
    def tau(self) -> torch.Tensor:
        return positive(self.raw_tau)

    @property
    def gamma(self) -> torch.Tensor:
        return positive(self.raw_gamma)

    def ap_params(self) -> Dict[str, torch.Tensor]:
        return {
            "b": self.b,
            "w": torch.exp(self.log_w),
            "ell": torch.exp(self.log_ell),
            "tau": self.tau,
            "gamma": self.gamma,
            "m_b": self.m_b,
            "s_b": self.s_b,
            "m_log_w": self.m_log_w,
            "s_log_w": self.s_log_w,
            "m_log_ell": self.m_log_ell,
            "s_log_ell": self.s_log_ell,
        }

    def log_likelihood_grid_for_ap(self, a: int) -> torch.Tensor:
        """
        Returns log p(y_a | z_a = j, continuous params) for all grid candidates j.

        Shape:
            [J_a]
        """
        ap = self.ap_data[a]

        y = ap.y                       # [n]
        outdoor = ap.outdoor           # [n]
        d2 = ap.d2_grid                # [n, J]

        b = self.b[a]
        w = torch.exp(self.log_w[a])
        ell = torch.exp(self.log_ell[a])
        tau = self.tau
        gamma = self.gamma

        kernel = torch.exp(
            -0.5 * d2 / ell.pow(2)
            - gamma * outdoor[:, None]
        )                              # [n, J]

        mean = b + w * kernel          # [n, J]
        resid = y[:, None] - mean      # [n, J]

        log_norm = -0.5 * math.log(2.0 * math.pi) - torch.log(tau)
        loglik = log_norm - 0.5 * resid.pow(2) / tau.pow(2)

        return loglik.sum(dim=0)       # [J]

    def negative_log_likelihood(self, hard_location: bool = False) -> torch.Tensor:
        """
        Marginalized location likelihood by default.

        hard_location=False:
            -sum_a logsumexp_j [log pi_aj + log p(y_a | z_a=j)]

        hard_location=True:
            -sum_a max_j [log pi_aj + log p(y_a | z_a=j)]
        """
        nll = torch.zeros((), device=self.device, dtype=self.dtype)

        for a, ap in enumerate(self.ap_data):
            loglik_j = self.log_likelihood_grid_for_ap(a)
            logpost_j = ap.logpi_grid + loglik_j

            if hard_location:
                loglik_a = torch.max(logpost_j)
            else:
                loglik_a = torch.logsumexp(logpost_j, dim=0)

            nll = nll - loglik_a

        return nll

    def negative_log_prior(self) -> torch.Tensor:
        """
        Prior penalties.

        These are written in the constrained/original parameterization.
        Constants are mostly irrelevant for MAP, but log(scale) terms are included
        where learned scales appear in Normal priors.
        """
        nlp = torch.zeros((), device=self.device, dtype=self.dtype)

        # AP-level hierarchical priors.
        nlp = nlp + normal_nlp(self.b, self.m_b, self.s_b).sum()
        nlp = nlp + normal_nlp(self.log_w, self.m_log_w, self.s_log_w).sum()
        nlp = nlp + normal_nlp(self.log_ell,
                               self.m_log_ell, self.s_log_ell).sum()

        # Hyperpriors.
        nlp = nlp + normal_nlp(self.m_b, -90.0, 15.0)
        nlp = nlp + halfnormal_nlp(self.s_b, 10.0)

        nlp = nlp + normal_nlp(self.m_log_w, math.log(30.0), 1.0)
        nlp = nlp + halfnormal_nlp(self.s_log_w, 1.0)

        nlp = nlp + normal_nlp(self.m_log_ell, math.log(15.0), 1.0)
        nlp = nlp + halfnormal_nlp(self.s_log_ell, 1.0)

        # Noise and outdoor attenuation.
        nlp = nlp + halfnormal_nlp(self.tau, 10.0)
        nlp = nlp + halfnormal_nlp(self.gamma, 1.0)

        return nlp

    def loss(self, hard_location: bool = False) -> torch.Tensor:
        return self.negative_log_likelihood(hard_location=hard_location) + self.negative_log_prior()

    @torch.no_grad()
    def location_responsibilities(self) -> List[torch.Tensor]:
        """
        Returns MAP-implied posterior-like responsibilities over grid locations.

        resp[a][j] approx proportional to:
            pi_aj * p(y_a | z_a=j, MAP continuous params)
        """
        out = []

        for a, ap in enumerate(self.ap_data):
            loglik_j = self.log_likelihood_grid_for_ap(a)
            logpost_j = ap.logpi_grid + loglik_j
            resp = torch.softmax(logpost_j, dim=0)
            out.append(resp.detach().cpu())

        return out

    @torch.no_grad()
    def summary(self) -> Dict[str, Any]:
        params = self.ap_params()
        responsibilities = self.location_responsibilities()

        b = params["b"].detach().cpu()
        w = params["w"].detach().cpu()
        ell = params["ell"].detach().cpu()

        out = {
            "tau": float(params["tau"].detach().cpu()),
            "gamma": float(params["gamma"].detach().cpu()),
            "rho_outdoor_retention": float(torch.exp(-params["gamma"]).detach().cpu()),
            "hyperparams": {
                "m_b": float(params["m_b"].detach().cpu()),
                "s_b": float(params["s_b"].detach().cpu()),
                "m_log_w": float(params["m_log_w"].detach().cpu()),
                "s_log_w": float(params["s_log_w"].detach().cpu()),
                "m_log_ell": float(params["m_log_ell"].detach().cpu()),
                "s_log_ell": float(params["s_log_ell"].detach().cpu()),
            },
            "aps": {},
        }

        for a, ap in enumerate(self.ap_data):
            resp = responsibilities[a]              # [J]
            grid_xy = ap.grid_xy.detach().cpu()     # [J, 2]

            j_map = int(torch.argmax(resp))
            mu_map = grid_xy[j_map]

            mu_mean = (resp[:, None] * grid_xy).sum(dim=0)

            out["aps"][ap.ap_id] = {
                "n_obs": len(ap.y),
                "b_map": float(b[a]),
                "w_map": float(w[a]),
                "ell_map": float(ell[a]),
                "mu_map_x": float(mu_map[0]),
                "mu_map_y": float(mu_map[1]),
                "mu_mean_x": float(mu_mean[0]),
                "mu_mean_y": float(mu_mean[1]),
                "location_prob_max": float(resp[j_map]),
                "location_argmax_index": j_map,
            }

        return out
