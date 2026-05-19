import numpy as np
from vlqbounds.models import TopPartnerSinglet
from vlqbounds.inference import VLQBounds
from example_support import run_parameter_scan, save_scan_outputs


def configure(bounds, mass: float, mixing: float) -> None:
    bounds.singletT_params(mT=mass, s_l=mixing)


def main() -> None:
    model = TopPartnerSinglet()
    bounds = VLQBounds(model)
    bounds.prepare_analysis()
    result_df = run_parameter_scan(
        bounds=bounds,
        parameter_callback=configure,
        masses=np.arange(600, 2001, 200),
        mixings=np.linspace(1e-5, 1.0, 10),
    )
    save_scan_outputs(
        result_df,
        output_stem="scan_t_singlet",
        x_label=r"$m_T\,\mathrm{[GeV]}$",
        title=r"$T\ \mathrm{Singlet}$",
        xlim=(500, 2000),
        ylim=(0.0, 0.5),
    )


if __name__ == "__main__":
    main()
