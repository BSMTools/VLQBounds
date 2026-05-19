import numpy as np
from vlqbounds.models import TopPartnerDoublet
from vlqbounds.inference import VLQBounds
from example_support import run_parameter_scan, save_scan_outputs


def configure(bounds, mass: float, mixing: float) -> None:
    bounds.doubletT_TB_params(mT=mass, s_u_r=mixing)


def main() -> None:
    model = TopPartnerDoublet()
    bounds = VLQBounds(model)
    bounds.prepare_analysis()
    result_df = run_parameter_scan(
        bounds=bounds,
        parameter_callback=configure,
        masses=np.arange(500, 2001, 200),
        mixings=np.linspace(1e-5, 1.0, 10),
    )
    save_scan_outputs(
        result_df,
        output_stem="scan_t_doublet_tb",
        x_label=r"$m_T\,\mathrm{[GeV]}$",
        title=r"$TB\ \mathrm{Doublet}$",
    )


if __name__ == "__main__":
    main()
