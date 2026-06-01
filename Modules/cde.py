import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchcde
except ImportError:
    torchcde = None


def _fill_forward(x, mask):
    if bool(torch.all(mask.bool())):
        return x
    b, t, c = x.shape
    lengths = mask.sum(dim=1).to(dtype=torch.long).clamp(min=1)
    last_idx = (lengths - 1).view(b, 1, 1).expand(b, 1, c)
    last = x.gather(dim=1, index=last_idx)
    pad = (torch.arange(t, device=x.device).view(1, t) >= lengths.view(b, 1)).unsqueeze(
        -1
    )
    return torch.where(pad, last.expand(b, t, c), x)


def _compute_tau(dt, mask):
    dt = dt * mask
    tau = torch.cumsum(dt, dim=1)
    b = dt.shape[0]
    last_valid = ((mask.sum(1).to(torch.long).clamp(min=1) - 1).clamp(min=0)).view(b, 1)
    denom = tau.gather(1, last_valid).clamp(min=1e-6)
    return tau / denom


class _CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.unet = UNet1D(
            in_channels=1,
            mid_channels=self.hidden_channels,
            out_channels=self.input_channels,
        )

    def forward(self, t, z):
        del t
        input_dtype = z.dtype
        z_in = z.unsqueeze(1).to(dtype=torch.float32)
        z_mask = torch.ones(
            (z_in.shape[0], 1, z_in.shape[-1]), device=z.device, dtype=z_in.dtype
        )
        vf = self.unet(z_in, z_mask)
        vf = vf.transpose(1, 2).contiguous()
        return vf.to(dtype=input_dtype)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, mask):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        return y * mask


class UNet1D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.mid_channels = int(mid_channels)
        groups = self._groups_for(self.mid_channels)
        self.in_block = ConvBlock1D(in_channels, self.mid_channels, groups=groups)
        self.down = nn.Conv1d(
            self.mid_channels, self.mid_channels, kernel_size=4, stride=2, padding=1
        )
        self.mid_block = ConvBlock1D(
            self.mid_channels, self.mid_channels, groups=groups
        )
        self.up = nn.ConvTranspose1d(
            self.mid_channels, self.mid_channels, kernel_size=4, stride=2, padding=1
        )
        self.out_block = ConvBlock1D(
            2 * self.mid_channels, self.mid_channels, groups=groups
        )
        self.proj = nn.Conv1d(self.mid_channels, out_channels, kernel_size=1)

    @staticmethod
    def _groups_for(channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x, mask):
        x0 = self.in_block(x, mask)
        d = self.down(x0)
        d_mask = F.interpolate(mask, size=d.shape[-1], mode="nearest")
        m = self.mid_block(d, d_mask)
        u = self.up(m)
        if u.shape[-1] != x0.shape[-1]:
            u = F.interpolate(u, size=x0.shape[-1], mode="nearest")
        h = torch.cat([x0, u], dim=1)
        h = self.out_block(h, mask)
        return self.proj(h) * mask


class NeuralCDE(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels=256,
        interpolation="linear",
        solver="reversible_heun",
        dt=1.0,
        atol=1e-5,
        rtol=1e-5,
    ):
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.input_channels = self.channels + 1
        self.interpolation = interpolation
        self.solver = solver
        self.dt = float(dt)
        self.atol = float(atol)
        self.rtol = float(rtol)

        self.initial_unet = UNet1D(
            in_channels=self.input_channels,
            mid_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
        )
        self.init_rf = 8
        self.func = _CDEFunc(self.input_channels, self.hidden_channels)
        self.readout_unet = UNet1D(
            in_channels=self.hidden_channels,
            mid_channels=self.hidden_channels,
            out_channels=self.channels,
        )

    def forward(self, x, mask, durations=None):
        if torchcde is None:
            raise ImportError("torchcde is required when model_params.cde.enabled=true")

        out_dtype = x.dtype
        compute_dtype = torch.float32

        mask_t = mask[:, 0, :].to(dtype=compute_dtype)
        x_t = x.transpose(1, 2).to(dtype=compute_dtype)
        b, t, _ = x_t.shape

        if durations is None:
            dt = torch.ones((b, t), device=x.device, dtype=compute_dtype)
        else:
            if durations.ndim == 3:
                durations = durations[:, 0, :]
            dt = durations.to(device=x.device, dtype=compute_dtype)

        tau = _compute_tau(dt, mask_t).unsqueeze(-1)
        path = torch.cat([x_t, tau], dim=-1)
        path = _fill_forward(path, mask_t)

        if self.interpolation == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(path)
            X = torchcde.LinearInterpolation(coeffs)
        else:
            coeffs = torchcde.natural_cubic_spline_coeffs(path)
            X = torchcde.NaturalCubicSpline(coeffs)

        rf = min(self.init_rf, path.shape[1])
        init_x = path[:, :rf, :].transpose(1, 2)
        init_mask = torch.ones((b, 1, rf), device=x.device, dtype=compute_dtype)
        init_feats = self.initial_unet(init_x, init_mask)
        z0 = init_feats[:, :, -1]
        t_grid = torch.linspace(
            float(X.interval[0]),
            float(X.interval[1]),
            t,
            device=x.device,
            dtype=compute_dtype,
        )

        kwargs = {
            "X": X,
            "z0": z0,
            "func": self.func,
            "t": t_grid,
            "method": self.solver,
            "atol": self.atol,
            "rtol": self.rtol,
        }
        if self.solver == "reversible_heun":
            kwargs["backend"] = "torchsde"
            kwargs["dt"] = self.dt
        else:
            kwargs["options"] = {"step_size": self.dt}

        z_t = torchcde.cdeint(**kwargs)
        if isinstance(z_t, tuple):
            z_t = z_t[0]
        y = self.readout_unet(z_t.transpose(1, 2), mask_t.unsqueeze(1))
        y = y * mask.to(dtype=y.dtype)
        return y.to(dtype=out_dtype)
