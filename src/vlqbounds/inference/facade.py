import datetime
from vlqbounds import __version__
from vlqbounds.models import *
from .evaluation import (
    append_scan_dataframe_row,
    sequence_is_missing_or_empty,
    validate_model_parameter_contract,
    validate_sine_parameter,
)
from vlqbounds.theory import constants as c
from .coupling_limits import CouplingLimitEngine


models = (TopPartnerSinglet, TopPartnerDoublet, TopPartnerPure, BottomPartnerSinglet, BottomPartnerDoublet, BottomPartnerPure, XPartnerDoublet, YPartnerDoublet, YPartnerTriplet)


# ══════════════════════════════════════════════════════════════════════════════
#  ANSI TERMINAL PALETTE
# ══════════════════════════════════════════════════════════════════════════════
class _C:
    """Centralised ANSI colour/style tokens."""
    RESET      = "\033[0m"
    BOLD       = "\033[1m"
    DIM        = "\033[2m"
    ITALIC     = "\033[3m"
    UNDERLINE  = "\033[4m"
    BLINK      = "\033[5m"

    # ── Foreground ──────────────────────────────────────────────────────────
    BLACK      = "\033[30m"
    RED        = "\033[31m"
    GREEN      = "\033[32m"
    YELLOW     = "\033[33m"
    BLUE       = "\033[34m"
    MAGENTA    = "\033[35m"
    CYAN       = "\033[36m"
    WHITE      = "\033[37m"

    # Bright foreground
    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    # ── Background ──────────────────────────────────────────────────────────
    BG_BLACK   = "\033[40m"
    BG_BLUE    = "\033[44m"
    BG_CYAN    = "\033[46m"

    # ── 256-colour helpers ───────────────────────────────────────────────────
    @staticmethod
    def fg256(n: int) -> str:
        return f"\033[38;5;{n}m"

    @staticmethod
    def bg256(n: int) -> str:
        return f"\033[48;5;{n}m"


def _c(style: str, text: str) -> str:
    """Wrap *text* in an ANSI style and reset."""
    return f"{style}{text}{_C.RESET}"


# ══════════════════════════════════════════════════════════════════════════════
#  DECORATIVE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_W = 62          # total printable width of the box

def _hr(char: str = "─", color: str = _C.CYAN) -> str:
    return _c(color, char * _W)


def _box_line(content: str, fill: str = " ", color: str = _C.CYAN,
              border_char: str = "║") -> str:
    """Return a bordered line padded to _W characters."""
    inner = _W - 2
    padded = content[:inner].ljust(inner, fill)
    border = _c(color + _C.BOLD, border_char)
    return f"{border}{padded}{border}"


def _kv(key: str, value, *, key_col: str = _C.BRIGHT_CYAN,
         val_col: str = _C.BRIGHT_WHITE, unit: str = "") -> str:
    """Format a key-value pair with optional unit."""
    k = _c(key_col + _C.BOLD, f"  {key:<28}")
    v = _c(val_col, f"{value}")
    u = _c(_C.DIM + _C.YELLOW, f" {unit}" if unit else "")
    return f"{k}{v}{u}"


def _section(title: str) -> str:
    """Sub-section header."""
    bar = _c(_C.fg256(208) + _C.BOLD, "▌")
    t   = _c(_C.BRIGHT_YELLOW + _C.BOLD + _C.UNDERLINE, f" {title} ")
    return f"\n{bar}{t}"


def _br() -> None:
    """Blank terminal line — acts as <BR>."""
    print()


def _result_badge(excluded: bool) -> str:
    if excluded:
        return (f"  {_c(_C.BG_BLACK + _C.BRIGHT_RED + _C.BOLD, '  ✖  EXCLUDED  ')}"
                f"  {_c(_C.RED, 'Signal exceeds experimental bound')}")
    else:
        return (f"  {_c(_C.BG_BLACK + _C.BRIGHT_GREEN + _C.BOLD, '  ✔  ALLOWED   ')}"
                f"  {_c(_C.GREEN, 'Consistent with current data')}")


def _decay_bar(fraction: float, width: int = 28) -> str:
    """
    Coloured ASCII progress bar representing a branching-ratio fraction ∈ [0,1].
    """
    filled = int(round(fraction * width))
    empty  = width - filled
    if fraction > 0.75:
        col = _C.BRIGHT_RED
    elif fraction > 0.40:
        col = _C.BRIGHT_YELLOW
    else:
        col = _C.BRIGHT_GREEN
    bar   = _c(col + _C.BOLD, "█" * filled) + _c(_C.DIM + _C.WHITE, "░" * empty)
    pct   = _c(_C.BRIGHT_WHITE + _C.BOLD, f" {fraction*100:5.1f}%")
    return f"[{bar}]{pct}"


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATE PARAMS
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════
class VLQBoundsFacade(CouplingLimitEngine):
    def __init__(self, m):
        self.first_run = True
        if isinstance(m, models):
            super().__init__(m)
            self.m = m
            self.df = c.df
            self.welcome_message()
            self.set_VLQ_type()

    # ── Analysis preparation ─────────────────────────────────────────────────
    def prepare_cross_section_inputs(self):
        self.initialize_tables_cms_and_atlas()
        self.all_processes()
        self.cs_dict()
        self.tb_xt_dict()

    def prepare_coupling_constraints(self):
        self.fill_coupling_tables()
        self.all_couplings()
        self.fill_sin_and_kappa()

    def prepare_analysis(self):
        self.initialize_tables_cms_and_atlas()
        self.cs_dict()
        self.tb_xt_dict()
        if not isinstance(self.m, (TopPartnerPure, BottomPartnerPure)):
            self.fill_coupling_tables()
            self.fill_sin_and_kappa()
            self.coupling_and_xs_info()

    # ── Parameter setters ────────────────────────────────────────────────────
    def singletT_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mT", {"k_T", "w_m", "s_l"})
        self.set_mass_and_coupling(kwargs, "mT", "k_T", "w_m", "s_l")

    def singletB_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mB", {"k_B", "w_m", "s_l"})
        self.set_mass_and_coupling(kwargs, "mB", "k_B", "w_m", "s_l")

    def doubletT_TB_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mT", {"k_T", "w_m", "s_u_r"})
        self.m.change_to_TB()
        self.set_mass_and_coupling(kwargs, "mT", "k_T", "w_m", "s_u_r")

    def doubletB_TB_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mB", {"k_B", "w_m", "s_d_r"})
        self.m.change_to_TB()
        self.set_mass_and_coupling(kwargs, "mB", "k_B", "w_m", "s_d_r")

    def doubletT_XT_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mT", {"k_T", "w_m", "s_r"})
        self.set_mass_and_coupling(kwargs, "mT", "k_T", "w_m", "s_r")

    def doubletX_XT_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mX", {"k_X", "w_m", "s_r"})
        self.set_mass_and_coupling(kwargs, "mX", "k_X", "w_m", "s_r")

    def doubletB_BY_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mB", {"k_B", "w_m", "s_r"})
        self.set_mass_and_coupling(kwargs, "mB", "k_B", "w_m", "s_r")

    def doubletY_BY_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mY", {"k_Y", "w_m", "s_r"})
        self.set_mass_and_coupling(kwargs, "mY", "k_Y", "w_m", "s_r")

    def tripletY_TBY_params(self, **kwargs):
        validate_model_parameter_contract(kwargs, "mY", {"k_Y", "w_m", "s_d_l"})
        self.set_mass_and_coupling(kwargs, "mY", "k_Y", "w_m", "s_d_l")

    # ── Pure-mode helpers ────────────────────────────────────────────────────
    def pure_T_to_Wb(self, mT: float) -> None:
        self.to_H, self.to_W, self.to_Z = False, True, False
        self.m.set_mT(mT)

    def pure_T_to_Zt(self, mT: float) -> None:
        self.to_H, self.to_W, self.to_Z = False, False, True
        self.m.set_mT(mT)

    def pure_T_to_Ht(self, mT: float) -> None:
        self.to_H, self.to_W, self.to_Z = True, False, False
        self.m.set_mT(mT)

    def pure_B_to_Wt(self, mB: float) -> None:
        self.to_H, self.to_W, self.to_Z = False, True, False
        self.m.set_mB(mB)

    def pure_B_to_Zb(self, mB: float) -> None:
        self.to_H, self.to_W, self.to_Z = False, False, True
        self.m.set_mB(mB)

    def pure_B_to_Hb(self, mB: float) -> None:
        self.to_H, self.to_W, self.to_Z = True, False, False
        self.m.set_mB(mB)

    # ── Public analysis entry-points ─────────────────────────────────────────
    def check_against_xs_limits(self):
        self.check_channel()
        self.print_result()
        i = self._channel
        self.data_frame(i)

    def check_against_coupling_limits(self):
        _, _, _, _ = self.coupling_limit()
        self.print_result()
        i = self._channel
        self.data_frame(i)

    def check_against_xs_and_coupling_limits(self):
        self.check_xs_and_coupling_limits()
        i = self._channel
        self.data_frame(i)

    def get_key(self):
        self.get_sensitive_limits_info()

    # ── VLQ type ─────────────────────────────────────────────────────────────
    def set_VLQ_type(self):
        if not isinstance(self.m, models):
            raise ValueError(_c(_C.BRIGHT_RED, "Error: this model is not included"))
        if isinstance(self.m, (BottomPartnerSinglet, BottomPartnerDoublet, BottomPartnerPure)):
            self.VLB = True
        elif isinstance(self.m, XPartnerDoublet):
            self.VLX = True
        elif isinstance(self.m, (YPartnerDoublet, YPartnerTriplet)):
            self.VLY = True

    # ── Internal helpers for display and physics summaries ───────────────────
    def _safe_model_call(self, *method_names: str):
        for method_name in method_names:
            method = getattr(self.m, method_name, None)
            if callable(method):
                try:
                    return method()
                except Exception:
                    continue
        return None

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _decay_widths_and_brs(self):
        if self.m.model() == 'Pure':
            if isinstance(self.m, TopPartnerPure):
                active = 'Wb' if self.to_W else 'Zt' if self.to_Z else 'Ht'
            elif isinstance(self.m, BottomPartnerPure):
                active = 'Wt' if self.to_W else 'Zb' if self.to_Z else 'Hb'
            else:
                active = 'Wt' if self.VLX else 'Wb'
            return self._safe_float(self._safe_model_call('get_width_mass_ratio')), [(active, 1.0)]

        width_ratio = self._safe_float(self._safe_model_call('get_width_mass_ratio'))

        if isinstance(self.m, TopPartnerSinglet):
            return width_ratio, [
                ('Wb', self._safe_float(self._safe_model_call('get_brTbw'))),
                ('Zt', self._safe_float(self._safe_model_call('get_brTzt'))),
                ('Ht', self._safe_float(self._safe_model_call('get_brTht'))),
            ]

        if isinstance(self.m, TopPartnerDoublet):
            if self.m.get_which_doublet() == 'XT':
                return width_ratio, [
                    ('Wb', self._safe_float(self._safe_model_call('get_brTbw_XT'))),
                    ('Zt', self._safe_float(self._safe_model_call('get_brTzt_XT'))),
                    ('Ht', self._safe_float(self._safe_model_call('get_brTht_XT'))),
                ]
            return width_ratio, [
                ('Wb', 0.0),
                ('Zt', self._safe_float(self._safe_model_call('get_brTzt_TB'))),
                ('Ht', self._safe_float(self._safe_model_call('get_brTht_TB'))),
            ]

        if isinstance(self.m, BottomPartnerSinglet):
            return width_ratio, [
                ('Wt', self._safe_float(self._safe_model_call('get_brBwt'))),
                ('Zb', self._safe_float(self._safe_model_call('get_brBzb'))),
                ('Hb', self._safe_float(self._safe_model_call('get_brBhb'))),
            ]

        if isinstance(self.m, BottomPartnerDoublet):
            if self.m.get_which_doublet() == 'BY':
                return width_ratio, [
                    ('Wt', self._safe_float(self._safe_model_call('get_brBwt_BY'))),
                    ('Zb', self._safe_float(self._safe_model_call('get_brBzb_BY'))),
                    ('Hb', self._safe_float(self._safe_model_call('get_brBhb_BY'))),
                ]
            return width_ratio, [
                ('Wt', 0.0),
                ('Zb', self._safe_float(self._safe_model_call('get_brBzb_TB'))),
                ('Hb', self._safe_float(self._safe_model_call('get_brBhb_TB'))),
            ]

        if isinstance(self.m, XPartnerDoublet):
            return width_ratio, [('Wt', 1.0)]
        if isinstance(self.m, (YPartnerDoublet, YPartnerTriplet)):
            return width_ratio, [('Wb', 1.0)]

        return width_ratio, []

    # ── Mass and coupling setter ──────────────────────────────────────────────
    def set_mass_and_coupling(self, kwargs, mass_key, coupling_key, width_key, sin_key):
        # Each call fully defines the point: drop any mixing, coupling or width
        # left over from a previous call with a different parameterisation.
        self.m._state.reset_coupling()
        if coupling_key in kwargs:
            if kwargs[coupling_key] is None:
                raise ValueError(_c(_C.BRIGHT_RED, f"Error: {coupling_key} must not be None"))
            self.m.set_coupling_strength(abs(kwargs[coupling_key]))

        elif width_key in kwargs:
            if kwargs[width_key] is None or kwargs[width_key] <= 0:
                raise ValueError(_c(_C.BRIGHT_RED, f"Error: {width_key} must be greater than 0"))
            self.m.set_width_mass_ratio(kwargs[width_key])

        elif sin_key in kwargs:
            if kwargs[sin_key] is None:
                raise ValueError(_c(_C.BRIGHT_RED, f"Error: {sin_key} must not be None"))
            validate_sine_parameter(kwargs[sin_key])
            if isinstance(self.m, (TopPartnerSinglet, BottomPartnerSinglet)):
                self.m.set_sin_l(kwargs[sin_key])
            elif isinstance(self.m, (TopPartnerDoublet, XPartnerDoublet)):
                if self.m.get_which_doublet() == 'XT':
                    self.m.set_sin_r(kwargs[sin_key])
                elif self.m.get_which_doublet() == 'TB':
                    self.m.set_sin_u_r(kwargs[sin_key])
            elif isinstance(self.m, (BottomPartnerDoublet, YPartnerDoublet)):
                if self.m.get_which_doublet() == 'BY':
                    self.m.set_sin_r(kwargs[sin_key])
                elif self.m.get_which_doublet() == 'TB':
                    self.m.set_sin_d_r(kwargs[sin_key])
            elif isinstance(self.m, YPartnerTriplet):
                self.m.set_sin_down_left(kwargs[sin_key])

        if isinstance(self.m, TopPartnerSinglet):
            self.m.set_mT(kwargs[mass_key])
            self.check_mass_range(kwargs[mass_key])
        elif isinstance(self.m, BottomPartnerSinglet):
            self.m.set_mB(kwargs[mass_key])
            self.check_mass_range(kwargs[mass_key], 'B')

        if isinstance(self.m, TopPartnerDoublet):
            self.m.set_mT(kwargs[mass_key])
            self.check_mass_range(kwargs[mass_key])
        elif isinstance(self.m, BottomPartnerDoublet):
            self.m.set_mB(kwargs[mass_key])
            self.check_mass_range(kwargs[mass_key], 'B')
        elif isinstance(self.m, XPartnerDoublet):
            self.m.set_mX(kwargs[mass_key])
            self.check_mass_range(kwargs[mass_key], 'X')
        elif isinstance(self.m, (YPartnerDoublet, YPartnerTriplet)):
            self.m.set_mY(kwargs[mass_key])
            self.check_mass_range(kwargs[mass_key], 'Y')

    # ── DataFrame helpers ────────────────────────────────────────────────────
    def set_df_attributes(self, mass, mixing, process, expt, luminosity,
                          energy, label, which_doublet, channel):
        self.df = append_scan_dataframe_row(
            self.df,
            mass=mass,
            mixing=mixing if not self.m.model() == 'Pure' else None,
            coupling=self.m.get_coupling_strength() if not self.m.model() == 'Pure' else None,
            width_ratio=self.m.get_width_mass_ratio() if not self.m.model() == 'Pure' else None,
            result=self._result,
            channel=channel,
            obs_ratio=self._obs_ratio,
            exp_ratio=self._exp_ratio,
            process=process[channel],
            experiment=expt[channel],
            luminosity=luminosity[channel],
            energy=energy[channel],
            label=label[channel],
            model=self.m.model(),
            which_doublet=which_doublet,
        )
        return self.df

    def data_frame(self, chan):
        def get_mixing():
            if self.m.model() == 'Singlet':
                return self.m.get_sin_left()
            if self.VLB:
                if self.m.model() == 'Doublet':
                    if self.m.get_which_doublet() == 'YB':
                        return self.m.get_sin_right()
                    else:
                        return self.m.get_sin_down_right()
            elif self.VLY:
                if self.m.model() == 'Doublet':
                    return self.m.get_sin_right()
                else:
                    return self.m.get_sin_down_left()
            elif self.VLX:
                if self.m.model() == 'Doublet':
                    return self.m.get_sin_right()
            else:
                if self.m.model() == 'Doublet':
                    if self.m.get_which_doublet() == 'TX':
                        return self.m.get_sin_right()
                    else:
                        return self.m.get_sin_up_right()

        def get_vlq_mass():
            if self.VLB:
                return self.m.get_mB()
            elif self.VLX:
                return self.m.get_mX()
            elif self.VLY:
                return self.m.get_mY()
            else:
                return self.m.get_mT()

        m_vlq        = get_vlq_mass()
        mixing       = get_mixing()
        which_doublet = self.m.get_which_doublet() if self.m.model() == 'Doublet' else None

        if not sequence_is_missing_or_empty(self.key) and sequence_is_missing_or_empty(self.coupling_key):
            self.df = self.set_df_attributes(m_vlq, mixing,
                                             self.process, self.expt,
                                             self.luminosity, self.energy,
                                             self.label, which_doublet, chan)
        elif sequence_is_missing_or_empty(self.key) and not sequence_is_missing_or_empty(self.coupling_key):
            self.df = self.set_df_attributes(m_vlq, mixing,
                                             self.coupling_process, self.coupling_expt,
                                             self.coupling_luminosity, self.coupling_energy,
                                             self.coupling_label, which_doublet, chan)
        elif not sequence_is_missing_or_empty(self.key) and not sequence_is_missing_or_empty(self.coupling_key):
            if len(self.key) <= chan:
                chan -= len(self.key)
                self.df = self.set_df_attributes(m_vlq, mixing,
                                                 self.coupling_process, self.coupling_expt,
                                                 self.coupling_luminosity, self.coupling_energy,
                                                 self.coupling_label, which_doublet, chan)
            else:
                self.df = self.set_df_attributes(m_vlq, mixing,
                                                 self.process, self.expt,
                                                 self.luminosity, self.energy,
                                                 self.label, which_doublet, chan)

    # ── Decay-to-mass ratio block ─────────────────────────────────────────────
    def _print_decay_to_mass(self):
        """Display Γ/M and the physically correct branching ratios from the active model."""
        width_ratio, decay_profile = self._decay_widths_and_brs()
        if width_ratio is None and not decay_profile:
            return

        print(_section("Decay-to-Mass Ratio  Γ/M"))
        _br()

        if width_ratio is not None:
            print(_kv("Γ/M  (total)", f"{width_ratio:.6f}", unit="[dimensionless]"))
        else:
            print(_kv("Γ/M  (total)", _c(_C.DIM, "N/A"), unit="[dimensionless]"))
        _br()

        if not decay_profile:
            print(_c(_C.DIM, "  [ no decay information available ]"))
            _br()
            return

        header = (
            f"  {_c(_C.DIM + _C.UNDERLINE, 'Channel'):<20}"
            f"{_c(_C.DIM + _C.UNDERLINE, 'BR'):>10}  "
            f"{_c(_C.DIM + _C.UNDERLINE, 'Visual bar')}"
        )
        print(header)
        print("  " + _c(_C.DIM, "─" * 54))

        colour_cycle = [_C.BRIGHT_BLUE, _C.BRIGHT_YELLOW, _C.BRIGHT_MAGENTA]
        valid_brs = []

        for idx, (label, br) in enumerate(decay_profile):
            colour = colour_cycle[idx % len(colour_cycle)]
            if br is None:
                row = (
                    f"  {_c(colour + _C.BOLD, label):<20}"
                    f"{_c(_C.DIM, '  N/A'):>10}  "
                    f"  {_c(_C.DIM, '[ no data ]')}"
                )
            else:
                valid_brs.append(float(br))
                row = (
                    f"  {_c(colour + _C.BOLD, label):<20}"
                    f"{_c(_C.BRIGHT_WHITE, f'{br:.4f}'):>10}  "
                    f"  {_decay_bar(br)}"
                )
            print(row)

        if valid_brs:
            total = sum(valid_brs)
            colour = _C.BRIGHT_GREEN if abs(total - 1.0) < 0.01 else _C.BRIGHT_RED
            print("  " + _c(_C.DIM, "─" * 54))
            print(f"  {_c(_C.DIM, 'Σ BR'):<20}{_c(colour + _C.BOLD, f'{total:.4f}'):>10}")

        _br()

    # ── Main print_result ─────────────────────────────────────────────────────
    def print_result(self):
        _br()
        print(_hr("═", _C.BRIGHT_CYAN))
        print(_box_line(
            _c(_C.BRIGHT_WHITE + _C.BOLD, f"  ⚛  VLQBounds  v{__version__}   │   Analysis Report"),
            color=_C.BRIGHT_CYAN,
        ))
        print(_hr("═", _C.BRIGHT_CYAN))

        # ── Model identity block ─────────────────────────────────────────────
        print(_section("Model & Kinematic Parameters"))
        _br()

        if self.VLB:
            print(_kv("VLQ type",      _c(_C.BRIGHT_MAGENTA + _C.BOLD, "Bottom-like  (B)")))
            print(_kv("Mass  mB",      f"{self.m.get_mB():.4f}", unit="GeV"))
            if self.m.model() != 'Pure':
                print(_kv("Relative width  Γ/M", f"{self.m.get_width_mass_ratio():.6f}"))
                print(_kv("Coupling  k_B",       f"{self.m.get_coupling_strength():.6f}"))
                if self.m.model() == 'Singlet':
                    print(_kv("Mixing angle  sin θ_L", f"{self.m.get_sin_left():.6f}"))
                elif self.m.model() == 'Doublet':
                    if self.m.get_which_doublet() == 'BY':
                        print(_kv("Mixing angle  sin θ_R", f"{self.m.get_sin_right():.6f}"))
                    else:
                        print(_kv("Mixing angle  sin θ_dR", f"{self.m.get_sin_down_right():.6f}"))

        elif self.VLX:
            print(_kv("VLQ type",      _c(_C.BRIGHT_BLUE + _C.BOLD,    "Exotic  X  (Q=5/3)")))
            print(_kv("Mass  mX",      f"{self.m.get_mX():.4f}", unit="GeV"))
            print(_kv("Relative width  Γ/M", f"{self.m.get_width_mass_ratio():.6f}"))
            print(_kv("Coupling  k_X",       f"{self.m.get_coupling_strength():.6f}"))
            print(_kv("Mixing angle  sin θ_R", f"{self.m.get_sin_right():.6f}"))

        elif self.VLY:
            print(_kv("VLQ type",      _c(_C.BRIGHT_YELLOW + _C.BOLD,  "Exotic  Y  (Q=−4/3)")))
            print(_kv("Mass  mY",      f"{self.m.get_mY():.4f}", unit="GeV"))
            print(_kv("Relative width  Γ/M", f"{self.m.get_width_mass_ratio():.6f}"))
            print(_kv("Coupling  k_Y",       f"{self.m.get_coupling_strength():.6f}"))
            if self.m.model() == 'Doublet':
                print(_kv("Mixing angle  sin θ_R", f"{self.m.get_sin_right():.6f}"))
            else:
                print(_kv("Mixing angle  sin θ_dL", f"{self.m.get_sin_down_left():.6f}"))

        else:
            print(_kv("VLQ type",      _c(_C.BRIGHT_GREEN + _C.BOLD,   "Top-like  (T)")))
            print(_kv("Mass  mT",      f"{self.m.get_mT():.4f}", unit="GeV"))
            if self.m.model() != 'Pure':
                print(_kv("Relative width  Γ/M", f"{self.m.get_width_mass_ratio():.6f}"))
                print(_kv("Coupling  k_T",       f"{self.m.get_coupling_strength():.6f}"))
                if self.m.model() == 'Singlet':
                    print(_kv("Mixing angle  sin θ_L", f"{self.m.get_sin_left():.6f}"))
                elif self.m.model() == 'Doublet':
                    if self.m.get_which_doublet() == 'XT':
                        print(_kv("Mixing angle  sin θ_R",  f"{self.m.get_sin_right():.6f}"))
                    else:
                        print(_kv("Mixing angle  sin θ_uR", f"{self.m.get_sin_up_right():.6f}"))

        print(_kv("Model representation", _c(_C.ITALIC + _C.CYAN, self.m.model())))
        _br()

        # ── Decay-to-mass ratio block ────────────────────────────────────────
        self._print_decay_to_mass()

        # ── Experiment info ──────────────────────────────────────────────────
        if self._channel != -1:
            print(_section("Experimental Reference"))
            _br()
            expt_name = None
            if not sequence_is_missing_or_empty(self.key) and not sequence_is_missing_or_empty(self.coupling_key):
                if self._channel < len(self.key):
                    expt_name = self.expt[self._channel]
                else:
                    expt_name = self.coupling_expt[self._channel - len(self.key)]
            elif not sequence_is_missing_or_empty(self.key):
                expt_name = self.expt[self._channel]
            elif not sequence_is_missing_or_empty(self.coupling_key):
                expt_name = self.coupling_expt[self._channel]

            if expt_name:
                print(_kv("Experiment", _c(_C.BRIGHT_WHITE + _C.BOLD, expt_name)))
            _br()

        # ── Bound result ─────────────────────────────────────────────────────
        print(_section("Limit Result"))
        _br()
        print(str(self))

        # Attempt to display excluded/allowed badge
        try:
            excluded = self._result
            print(_result_badge(excluded))
        except AttributeError:
            pass

        if hasattr(self, '_obs_ratio') and self._obs_ratio is not None:
            _br()
            print(_kv("Observed  σ/σ_lim",  f"{self._obs_ratio:.4f}"))
        if hasattr(self, '_exp_ratio') and self._exp_ratio is not None:
            print(_kv("Expected  σ/σ_lim",  f"{self._exp_ratio:.4f}"))

        _br()
        print(_hr("═", _C.BRIGHT_CYAN))
        _br()

    # ── Mass-range warning ────────────────────────────────────────────────────
    def check_mass_range(self, m, vlq='T'):
        key, M_vlq = '', -1

        def _resolve(obs_key, obs_data, coupling_obs):
            nonlocal key, M_vlq
            if not sequence_is_missing_or_empty(self.coupling_key):
                if sequence_is_missing_or_empty(self.key):
                    M_vlq = coupling_obs
                    key   = self.coupling_key
                else:
                    key   = self.key + self.coupling_key
                    M_vlq = obs_data + coupling_obs
            else:
                if sequence_is_missing_or_empty(self.key):
                    raise Exception(_c(_C.BRIGHT_RED, "Error: experimental data are not filled"))
                key   = self.key
                M_vlq = obs_data

        if   vlq == 'T': _resolve(None, self.MT,   self.coupling_MT_obs)
        elif vlq == 'B': _resolve(None, self.MB,   self.coupling_MB_obs)
        elif vlq == 'X': _resolve(None, self.MX,   self.coupling_MX_obs)
        elif vlq == 'Y': _resolve(None, self.MY,   self.coupling_MY_obs)

        mini = min(min(M_vlq[i]) for i in range(len(key)))
        maxi = max(max(M_vlq[i]) for i in range(len(key)))

        if m < mini or m > maxi:
            warn = (
                f"\n  {_c(_C.BRIGHT_YELLOW + _C.BOLD, '⚠  MASS EXTRAPOLATION WARNING')}\n"
                f"  {_c(_C.YELLOW, f'Requested mass {m:.1f} GeV lies outside')}\n"
                f"  {_c(_C.YELLOW, f'experimental range  [{mini:.1f}, {maxi:.1f}] GeV')}\n"
                f"  {_c(_C.DIM,    'Results may be unreliable beyond the covered interval.')}"
            )
            print(warn)


    # ── Welcome banner ────────────────────────────────────────────────────────
    def welcome_message(self):
        self.first_run = False
        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

        gold = _C.fg256(220) + _C.BOLD
        teal = _C.fg256(51) + _C.BOLD
        silver = _C.fg256(250)
        dim = _C.DIM

        top = _c(gold, "╔" + "═" * (_W - 2) + "╗")
        bot = _c(gold, "╚" + "═" * (_W - 2) + "╝")
        mid = _c(gold, "╠" + "═" * (_W - 2) + "╣")

        def _bline(txt: str, col: str = silver) -> str:
            inner = _W - 4
            padded = txt[:inner].center(inner)
            return _c(gold, "║ ") + _c(col, padded) + _c(gold, " ║")

        print()
        print(top)
        print(_bline(""))
        print(_bline(f"⚛  VLQBounds   v{__version__}  ⚛", gold))
        print(_bline(""))
        print(mid)
        print(_bline("Python toolkit for confronting vector-like quark scenarios with collider limits", teal))
        print(_bline(""))
        print(_bline("Authors:", dim))
        print(_bline("Abdesslam Arhrib  ·  Rachid Benbrik  ·  Mohammed Boukidi", silver))
        print(_bline("Mohamed Ech-chaouy  ·  Stefano Moretti", silver))
        print(_bline("Khawla Salime  ·  Qi-Shu Yan", silver))
        print(_bline("arXiv:2605.20848", dim))
        print(_bline(""))
        print(mid)
        print(_bline(f"Session started  {ts}", dim))
        print(_bline(""))
        print(bot)
        _br()


VLQBounds = VLQBoundsFacade
