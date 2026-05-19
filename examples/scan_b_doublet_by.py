import numpy as np
from vlqbounds.models import BottomPartnerDoublet
from vlqbounds.inference import VLQBounds
from example_support import run_parameter_scan, save_scan_outputs


def configure(bounds, mass: float, mixing: float) -> None:
    bounds.doubletB_BY_params(mB=mass, s_r=mixing)


def main() -> None:
    model = BottomPartnerDoublet()
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
        output_stem="scan_b_doublet_by",
        x_label=r"$m_B\,\mathrm{[GeV]}$",
        title=r"$BY\ \mathrm{Doublet}$",
    )


if __name__ == "__main__":
    main()
