from vlqbounds.theory import mass_ratio, compute_limit_ratios
from vlqbounds.models import TopPartnerSinglet, BottomPartnerDoublet, XPartnerDoublet, YPartnerDoublet, YPartnerTriplet
from vlqbounds.inference import VLQBounds
from vlqbounds.constraints import ExperimentalBoundsRepository


def main() -> None:
    assert abs(mass_ratio(10.0, 2.0) - 5.0) < 1e-12
    obs_ratio, exp_ratio = compute_limit_ratios(2.0, 4.0, 8.0)
    assert abs(obs_ratio - 0.5) < 1e-12
    assert abs(exp_ratio - 0.25) < 1e-12
    for model in (TopPartnerSinglet(), BottomPartnerDoublet(), XPartnerDoublet(), YPartnerDoublet(), YPartnerTriplet()):
        _ = VLQBounds(model)
    _ = ExperimentalBoundsRepository(TopPartnerSinglet())
    print("Import validation passed.")


if __name__ == "__main__":
    main()
