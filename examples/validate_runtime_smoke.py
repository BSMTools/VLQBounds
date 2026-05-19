from vlqbounds.models import TopPartnerSinglet, BottomPartnerDoublet, XPartnerDoublet
from vlqbounds.inference import VLQBounds


def run_singlet_t() -> None:
    bounds = VLQBounds(TopPartnerSinglet())
    bounds.prepare_analysis()
    bounds.singletT_params(mT=1000.0, s_l=0.05)
    bounds.check_against_xs_and_coupling_limits()
    assert bounds.df is not None


def run_doublet_b() -> None:
    bounds = VLQBounds(BottomPartnerDoublet())
    bounds.prepare_analysis()
    bounds.doubletB_BY_params(mB=1200.0, s_r=0.10)
    bounds.check_against_xs_and_coupling_limits()
    assert bounds.df is not None


def run_doublet_x() -> None:
    bounds = VLQBounds(XPartnerDoublet())
    bounds.prepare_analysis()
    bounds.doubletX_XT_params(mX=1200.0, s_r=0.10)
    bounds.check_against_xs_and_coupling_limits()
    assert bounds.df is not None


def main() -> None:
    run_singlet_t()
    run_doublet_b()
    run_doublet_x()
    print("Runtime smoke validation passed.")


if __name__ == "__main__":
    main()
