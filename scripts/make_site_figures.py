"""Export the two data-driven figures used on the portfolio write-up, as inline SVG.

The site renders report figures as *inline* SVG styled by its own CSS variables, so they
follow the light/dark theme. A PNG cannot do that (and a matplotlib export would drag a
white background into dark mode), so we emit SVG paths that use the site's existing classes:

    spec-trace   the observed mixture           (ink)
    spec-comp    individual clean components    (muted, thin)
    spec-recon   model output in masked spans   (accent - reserved for exactly this)
    spec-axis / spec-tick

Figure 1: one mixture drawn over the clean components summed into it.
Figure 2: a real masked reconstruction from the pretrained model.

Figure 2 needs the reconstruction head, not just the encoder, so it loads
`experiments/pretrained_encoder_full.pt` written by scripts/pretrain.py.

    python scripts/make_site_figures.py --out <dir>

Then paste each file's contents into the <figure> in the site's page. The page carries a
comment pointing back here, so the provenance of the curves is not a mystery later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from spectral.data.generator import MixtureGenerator
from spectral.data.library import CompoundLibrary
from spectral.models.masked import MaskedSpectralModel, make_batch_mask
from spectral.training.config import PretrainExperimentConfig
from spectral.utils import PROJECT_ROOT

W, H = 1000.0, 200.0     # viewBox; matches the site's hero spectrum band
PAD_X, BASE_Y, TOP_Y = 8.0, 168.0, 16.0


def _scale(y: np.ndarray, ymax: float) -> np.ndarray:
    """Map intensity -> SVG y (inverted, baseline at BASE_Y)."""
    return BASE_Y - (y / ymax) * (BASE_Y - TOP_Y)


def _xs(n: int) -> np.ndarray:
    return np.linspace(PAD_X, W - PAD_X, n)


def _simplify(xs: np.ndarray, ys: np.ndarray, eps: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Ramer-Douglas-Peucker in SVG units.

    A clean component trace is mostly flat baseline, and emitting 2048 points each makes the
    page heavy for no visual gain. RDP drops points lying within `eps` PERPENDICULAR distance
    of the chord they sit on, so flat stretches collapse while peak apexes are kept exactly.

    Note the bound is perpendicular, not vertical: on a near-vertical peak edge a point can
    move slightly along the edge. That is invisible at render size, and the noisy observed
    trace barely simplifies at all (its wiggle is real signal to draw), so most of the saving
    comes from the smooth components.
    """
    pts = np.column_stack([xs, ys])
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = pts[j] - pts[i]
        norm = np.hypot(*seg)
        rel = pts[i + 1 : j] - pts[i]
        if norm == 0:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            # perpendicular distance to the chord i->j, via the 2D cross product
            # (np.cross no longer takes 2D vectors in numpy 2.x)
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm
        k = int(np.argmax(d))
        if d[k] > eps:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return pts[keep, 0], pts[keep, 1]


def _path(xs: np.ndarray, ys: np.ndarray, step: int = 1) -> str:
    xs, ys = xs[::step], ys[::step]
    if len(xs) > 2:
        xs, ys = _simplify(xs, ys)
    pts = [f"{x:.1f} {y:.1f}" for x, y in zip(xs, ys)]
    return "M" + " L".join(pts)


def _ticks(ppm_min: float, ppm_max: float, y: float = 192.0) -> str:
    """ppm axis, high shift on the left as NMR is conventionally drawn.

    The unit rides on the last tick ("0 ppm") rather than sitting as a separate right-anchored
    label: at x = W - PAD_X both would land on the same point and overprint each other.
    """
    out = []
    for ppm in (8, 6, 4, 2):
        frac = (ppm_max - ppm) / (ppm_max - ppm_min)
        x = PAD_X + frac * (W - 2 * PAD_X)
        out.append(f'<text class="spec-tick" x="{x:.1f}" y="{y:.1f}" text-anchor="middle">{ppm}</text>')
    out.append(f'<text class="spec-tick" x="{W - PAD_X:.1f}" y="{y:.1f}" text-anchor="end">0 ppm</text>')
    return "".join(out)


def figure_mixture(cfg, library, index: int = 2) -> str:
    """Mixture stacked above its clean components.

    Overlaying the components under the mixture does not read: the observed trace is noisy and
    swamps them. A stack plot - the conventional way to show a decomposition in NMR - makes the
    "this is the sum of those" relationship legible. All components share one vertical scale, so
    their relative heights still carry the concentration differences.
    """
    gen = MixtureGenerator(cfg.data, library)
    s = gen.generate(index)
    xs = _xs(s.mixture.shape[0])

    rows = [r for r in s.components if r.max() > 0]
    k = len(rows)
    lane_h, first_lane = 34.0, 150.0
    height = first_lane + (k - 1) * lane_h + 30.0

    mix_base, mix_top = 104.0, 14.0
    mix_y = mix_base - (s.mixture / float(s.mixture.max())) * (mix_base - mix_top)

    comp_max = max(float(r.max()) for r in rows)   # shared scale keeps concentrations comparable
    lanes = []
    for i, row in enumerate(rows):
        base = first_lane + i * lane_h
        y = base - (row / comp_max) * 26.0
        lanes.append(f'  <line class="spec-axis" x1="{PAD_X}" y1="{base:.1f}" x2="{W - PAD_X}" y2="{base:.1f}"/>')
        lanes.append(f'  <path class="spec-comp" d="{_path(xs, y, step=2)}"/>')

    return f'''<svg viewBox="0 0 {W:.0f} {height:.0f}" role="img" aria-labelledby="f1t f1d">
  <title id="f1t">A synthetic mixture above the components summed into it</title>
  <desc id="f1d">The observed mixture on top, noisy and overlapping. Beneath it, the {k} clean
  component signals that were summed to make it, each on its own lane and sharing one vertical
  scale so their relative concentrations are comparable. Every component is known exactly by
  construction, which is what makes the disentangling question measurable.</desc>
  <text class="spec-tick" x="{PAD_X}" y="12">mixture</text>
  <line class="spec-axis" x1="{PAD_X}" y1="{mix_base}" x2="{W - PAD_X}" y2="{mix_base}"/>
  <path class="spec-trace" d="{_path(xs, mix_y)}"/>
  <text class="spec-tick" x="{PAD_X}" y="130">components ({k}), same vertical scale</text>
{chr(10).join(lanes)}
  {_ticks(cfg.data.grid.ppm_min, cfg.data.grid.ppm_max, y=height - 6)}
</svg>'''


def figure_reconstruction(cfg, library, device, index: int = 5) -> str:
    """A real masked reconstruction: what the model fills into spans it cannot see."""
    full_path = PROJECT_ROOT / "experiments" / "pretrained_encoder_full.pt"
    if not full_path.exists():
        raise SystemExit(
            f"missing {full_path}\nRun:  python scripts/pretrain.py --config configs/pretrain.yaml"
        )

    model = MaskedSpectralModel(cfg.model, cfg.data.grid.n_points).to(device)
    model.load_state_dict(torch.load(full_path, map_location=device), strict=True)
    model.eval()

    gen = MixtureGenerator(cfg.data, library)
    x = torch.from_numpy(gen.generate(index).mixture.astype(np.float32))[None].to(device)
    mask = make_batch_mask(1, model.n_patches, cfg.pretrain.mask_ratio, cfg.pretrain.span_len,
                           np.random.default_rng(7)).to(device)
    with torch.no_grad():
        recon = model(x, mask)[0].cpu().numpy()          # (P, patch_size)
    truth = model.to_patches(x.cpu(), model.patch_size)[0].numpy()
    m = mask[0].cpu().numpy()

    ymax = float(truth.max())
    flat_truth = truth.reshape(-1)
    xs = _xs(flat_truth.shape[0])
    ps = model.patch_size

    # Visible input: the model sees everything except the masked spans.
    visible = truth.copy()
    visible[m] = np.nan
    # Reconstruction is only meaningful where the model was blind.
    recon_masked = np.full_like(recon, np.nan)
    recon_masked[m] = recon[m]

    def segments(flat: np.ndarray) -> str:
        """Break a NaN-gapped series into separate paths so gaps are not bridged."""
        ys = _scale(flat, ymax)
        out, run_x, run_y = [], [], []
        for x_i, y_i, v in zip(xs, ys, flat):
            if np.isnan(v):
                if len(run_x) > 1:
                    out.append((list(run_x), list(run_y)))
                run_x, run_y = [], []
            else:
                run_x.append(x_i); run_y.append(y_i)
        if len(run_x) > 1:
            out.append((run_x, run_y))
        return out

    vis = "\n".join(f'  <path class="spec-trace" d="{_path(np.array(rx), np.array(ry))}"/>'
                    for rx, ry in segments(visible.reshape(-1)))
    rec = "\n".join(f'  <path class="spec-recon" d="{_path(np.array(rx), np.array(ry))}"/>'
                    for rx, ry in segments(recon_masked.reshape(-1)))

    # Faint band behind each hidden span, so "the model saw nothing here" is visible.
    bands = []
    for p in np.flatnonzero(m):
        x0 = PAD_X + (p * ps) / flat_truth.shape[0] * (W - 2 * PAD_X)
        wpx = ps / flat_truth.shape[0] * (W - 2 * PAD_X)
        bands.append(f'  <rect class="spec-hidden" x="{x0:.1f}" y="0" width="{wpx:.1f}" height="{H:.0f}"/>')
    n_masked = int(m.sum())

    return f'''<svg viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-labelledby="f2t f2d">
  <title id="f2t">A masked span reconstructed by the pretrained model</title>
  <desc id="f2d">{n_masked} of {model.n_patches} patches are hidden (shaded bands). The model
  fills them in (accent) close to the hidden truth, and returns a denoised curve rather than
  the noisy measurement it was trained on.</desc>
  <line class="spec-axis" x1="{PAD_X}" y1="{BASE_Y}" x2="{W - PAD_X}" y2="{BASE_Y}"/>
{chr(10).join(bands)}
  <path class="spec-comp" d="{_path(xs, _scale(flat_truth, ymax))}"/>
{vis}
{rec}
  {_ticks(cfg.data.grid.ppm_min, cfg.data.grid.ppm_max)}
</svg>'''


def main() -> None:
    ap = argparse.ArgumentParser(description="Export inline SVG figures for the website.")
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--out", default="experiments/site")
    args = ap.parse_args()

    cfg = PretrainExperimentConfig.from_yaml(args.config)
    library = CompoundLibrary.from_config(cfg.data.library)
    device = torch.device("cpu")

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    (out / "fig1_mixture.svg").write_text(figure_mixture(cfg, library))
    print(f"wrote {out / 'fig1_mixture.svg'}")
    (out / "fig2_reconstruction.svg").write_text(figure_reconstruction(cfg, library, device))
    print(f"wrote {out / 'fig2_reconstruction.svg'}")
    print("\nPaste each into the matching <figure> on the site page.")


if __name__ == "__main__":
    main()
