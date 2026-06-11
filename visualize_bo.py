#!/usr/bin/env python3
"""
visualize_bo.py

Genera gráficos de análisis para los estudios Optuna de la búsqueda bayesiana.

Uso:
    # Un estudio específico, se abre en el navegador:
    python visualize_bo.py bo_results/20260604_150310/snn_bo_SNSGAII_ds1.db

    # Todos los estudios de un directorio, guarda HTML:
    python visualize_bo.py bo_results/20260604_150310/ --save

    # Comparar todos los estudios en un solo reporte:
    python visualize_bo.py bo_results/20260604_150310/ --save --report
"""

import argparse
import sys
import webbrowser
from pathlib import Path


def load_study(db_path: Path):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = db_path.stem
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")


def make_plots(study, save_dir: Path | None = None, open_browser: bool = True):
    import optuna.visualization as vis

    study_name = study.study_name
    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")

    print(f"\n  {study_name}: {n_complete} trials completados, best HV={study.best_value:.6f}")

    plots = {
        "optimization_history":  vis.plot_optimization_history(study),
        "param_importances":     vis.plot_param_importances(study),
        "parallel_coordinate":   vis.plot_parallel_coordinate(study),
        "contour":               vis.plot_contour(study),
        "slice":                 vis.plot_slice(study),
    }

    for name, fig in plots.items():
        fig.update_layout(title=f"{study_name} — {name.replace('_', ' ').title()}")
        if save_dir:
            out = save_dir / f"{study_name}_{name}.html"
            fig.write_html(str(out))
            print(f"    guardado: {out}")
        if open_browser:
            fig.show()


def make_summary_report(studies: list, save_dir: Path):
    """Genera un HTML con tabla resumen comparando todos los estudios."""
    rows = []
    for study in studies:
        bt = study.best_trial
        arch = bt.params.get("architecture", "?")
        rows.append({
            "study":        study.study_name,
            "best_hv":      f"{study.best_value:.6f}",
            "trials_ok":    sum(1 for t in study.trials if t.state.name == "COMPLETE"),
            "trials_total": len(study.trials),
            "architecture": arch,
            "encoder":      bt.params.get("encoder", ["poisson","poisson","ttfs","ttfs"][int(arch)-1] if isinstance(arch, int) else "?"),
            "best_trial_n": bt.number,
        })

    rows.sort(key=lambda r: float(r["best_hv"]), reverse=True)

    html = ["<html><head><meta charset='utf-8'>",
            "<style>body{font-family:monospace;padding:20px}",
            "table{border-collapse:collapse;width:100%}",
            "th,td{border:1px solid #ccc;padding:8px 12px;text-align:left}",
            "th{background:#f0f0f0}tr:hover{background:#fafafa}</style></head><body>",
            "<h2>Resumen de estudios BO — SparseSNN</h2>",
            "<table><tr><th>Estudio</th><th>Best HV</th><th>Arch</th>",
            "<th>Trials OK/Total</th><th>Best trial #</th></tr>"]

    for r in rows:
        html.append(
            f"<tr><td>{r['study']}</td><td><b>{r['best_hv']}</b></td>"
            f"<td>{r['architecture']}</td>"
            f"<td>{r['trials_ok']}/{r['trials_total']}</td>"
            f"<td>{r['best_trial_n']}</td></tr>"
        )

    html.append("</table></body></html>")

    out = save_dir / "summary_report.html"
    out.write_text("\n".join(html))
    print(f"\n  Reporte resumen guardado: {out}")
    webbrowser.open(str(out.resolve()))


def main():
    parser = argparse.ArgumentParser(description="Visualiza estudios Optuna de la búsqueda BO")
    parser.add_argument("target", help="Archivo .db o directorio con .db files")
    parser.add_argument("--save",   action="store_true",
                        help="Guarda los gráficos como HTML en un subdirectorio 'plots/'")
    parser.add_argument("--no-browser", action="store_true",
                        help="No abre el navegador (útil si sólo querés guardar)")
    parser.add_argument("--report", action="store_true",
                        help="Genera un HTML resumen comparando todos los estudios (requiere --save)")
    parser.add_argument("--study",  default=None,
                        help="Filtra por nombre de estudio (ej. snn_bo_SNSGAII_ds2)")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: pip install optuna", file=sys.stderr)
        sys.exit(1)

    target = Path(args.target)
    if target.is_file():
        db_files = [target]
    elif target.is_dir():
        db_files = sorted(target.glob("*.db"))
    else:
        print(f"ERROR: '{target}' no existe", file=sys.stderr)
        sys.exit(1)

    if args.study:
        db_files = [f for f in db_files if args.study in f.stem]

    if not db_files:
        print("No se encontraron archivos .db", file=sys.stderr)
        sys.exit(1)

    save_dir = None
    if args.save:
        save_dir = target if target.is_dir() else target.parent
        save_dir = save_dir / "plots"
        save_dir.mkdir(exist_ok=True)
        print(f"Guardando gráficos en: {save_dir}")

    open_browser = not args.no_browser

    studies = []
    for db in db_files:
        try:
            study = load_study(db)
            studies.append(study)
            make_plots(study, save_dir=save_dir, open_browser=open_browser and not args.report)
        except Exception as e:
            print(f"  SKIP {db.stem}: {e}", file=sys.stderr)

    if args.report and studies:
        if not save_dir:
            save_dir = target if target.is_dir() else target.parent
            save_dir.mkdir(exist_ok=True)
        make_summary_report(studies, save_dir)


if __name__ == "__main__":
    main()
