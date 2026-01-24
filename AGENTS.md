# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI entry for `train`, `eval`, `visualize`.
- `config/*.yml`: Dataset-specific hyperparameters (e.g., `2016.10a.yml`).
- `data_loader/`: Dataset IO, splits, and DataLoader utilities.
- `models/`: AWN model (`model.py`) and lifting scheme (`lifting.py`).
- `util/`: Training loop, evaluation, logging, plotting, config helpers.
- `assets/`: Architecture and visualization images.
- `data/`, `checkpoint/`, `training/`, `inference/`: Local data and run artifacts (gitignored).

## Build, Test, and Development Commands
- Environment: Python ≥ 3.6; PyTorch ≥ 1.7 (tested 1.8.1).
- Install deps (example): `pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm h5py pyyaml torchattacks`
- Train: `python main.py --mode train --dataset 2016.10a`
- Evaluate: `python main.py --mode eval --dataset 2016.10a --ckpt_path ./checkpoint`
- Visualize: `python main.py --mode visualize --dataset 2016.10a`
- Adversarial eval: `python main.py --mode adv_eval --dataset 2016.10a --ckpt_path ./checkpoint --attack cw`
- Multi-attack eval: `python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint --attack_list fgsm,pgd,cw --plot_freq`
- Device: `--device cuda` or `--device cpu` (e.g., `CUDA_VISIBLE_DEVICES=0`).

## Coding Style & Naming Conventions
- Python with 4-space indentation; prefer snake_case for new code; preserve existing public names (e.g., `Create_Data_Loader`).
- Line length ≤ 100; add docstrings and type hints where helpful.
- Config lives in `config/*.yml`; new fields must be optional and documented in `README.md`.
- Don’t break existing CLI flags; add argparse options with sensible defaults.
- Formatting (optional): `black . && isort .` if available; no strict linter is enforced in repo.

## Testing Guidelines
- No formal unit tests yet. Validate via `--mode eval` and review metrics/logs in `training/<DATASET>_*/result`.
- If adding tests, use `pytest`; place files in `tests/` named `test_*.py`. Start with `util/` helpers and small, deterministic cases.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., "Add AWN eval plot"); group related changes.
- PRs must include: purpose/context, exact commands to reproduce (train/eval), dataset used, before/after metrics, and artifact paths (e.g., `training/2016.10a_*/result/acc.svg`).
- Do not commit datasets or checkpoints. Keep large files in `data/` and `checkpoint/`.

## Security & Configuration Tips
- Keep datasets local and respect licenses. Use `--seed` for reproducibility.
- Adding a new dataset: create `config/<name>.yml`, ensure label mappings align with `util/config.py` and `data_loader` expectations.
