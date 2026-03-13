# LD23 Simplified Reproduction

纯 AI 代码，无质量保证。

This workspace contains a compact, runnable prototype inspired by `LD23.pdf`:

- 3D two-phase fluid with a coarse `D3Q27` flow grid and a fine `D3Q7` phase grid
- a rigid `d20` drop scene and a scripted airplane skim scene
- pause, reset, scene switching, and orbit camera controls

The implementation intentionally stays conservative compared to the paper:

- one rigid body only
- analytic tank and `d20` geometry
- moderate density ratio and viscosity settings for stability on a home PC

## Environment

Create and use the local virtual environment only:

```powershell
C:\Python312\python.exe -m venv .venv
F:\simulation\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
F:\simulation\.venv\Scripts\python.exe -m sim.app --preset d20_drop --quality default --backend cuda
F:\simulation\.venv\Scripts\python.exe -m sim.app --preset plane_skim --quality default --backend cuda
```

Useful options:

```powershell
F:\simulation\.venv\Scripts\python.exe -m sim.app --quality low --backend cpu
F:\simulation\.venv\Scripts\python.exe -m sim.app --headless --steps 600
```

## Controls

- `Space`: pause / resume
- `R`: reset scene
- `Tab`: next scene
- `1`: switch to `d20_drop`
- `2`: switch to `plane_skim`
- `Esc`: quit
- Left mouse drag: orbit camera
- Mouse wheel: zoom

## Tests

```powershell
F:\simulation\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The smoke test runs the solver in CPU mode on a small grid and only checks for finite values and bounded phase fields.
