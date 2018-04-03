from types import SimpleNamespace

from models.gpr.variables.mg import eos_text_to_code
from models.gpr.variables.state import temperature


class EOS_params():
    def __init__(self, EOS, ρ0, cv, p0, Tref,
                 γ, pINF, c0, Γ0, s):

        self.EOS = eos_text_to_code(EOS)

        self.ρ0 = ρ0
        self.cv = cv
        self.p0 = p0
        self.Tref = Tref

        if EOS == 'sg':
            self.γ = γ
            self.pINF = pINF

        elif EOS == 'smg':
            self.Γ0 = Γ0
            self.c02 = c0**2
            self.s = s


def params(MP, Rc, EOS, THERMAL,
           ρ0, p0, Tref, T0, cv,
           γ, pINF, c0, Γ0, s, e0,
           cs, τ1, μ, σY, n, PLASTIC,
           cα, τ2):

    MP.Rc = Rc
    MP.EOS = eos_text_to_code(EOS)
    MP.THERMAL = THERMAL

    MP.ρ0 = ρ0
    MP.p0 = p0
    MP.Tref = Tref
    MP.T0 = T0
    MP.cv = cv

    if EOS == 'sg':
        MP.γ = γ
        MP.pINF = pINF

    elif EOS == 'smg':
        MP.Γ0 = Γ0
        MP.c02 = c0**2
        MP.s = s
        MP.e0 = e0

    MP.cs2 = cs**2
    MP.τ1 = τ1
    MP.PLASTIC = PLASTIC
    if PLASTIC:
        MP.σY = σY
        MP.n = n

    if cα is not None:
        MP.cα2 = cα**2
        MP.τ2 = τ2


def material_parameters(EOS, ρ0, cv, p0, cs,
                        Tref=None, γ=None, pINF=None,
                        c0=None, Γ0=None, s=None, e0=None,
                        μ=None, τ1=None, σY=None, n=None, PLASTIC=False,
                        cα=None, κ=None, Pr=None,
                        Rc=8.31445985):
    """ An object to hold the material constants
    """
    assert(EOS in ['sg', 'smg', 'gr'])

    if Tref is None:
        Tref = 0

    if (γ is not None) and (pINF is None):
        pINF = 0

    P = EOS_params(EOS, ρ0, cv, p0, Tref, γ, pINF, c0, Γ0, s)
    T0 = temperature(ρ0, p0, P)

    if (not PLASTIC) and (τ1 is None):
        τ1 = 6 * μ / (ρ0 * cs**2)

    if cα is not None:
        THERMAL = True
        if Pr is not None:
            κ = μ * γ * cv / Pr
        τ2 = κ * ρ0 / (T0 * cα**2)
    else:
        THERMAL = False
        τ2 = None

    MP = SimpleNamespace()

    params(MP, Rc, EOS, THERMAL,
           ρ0, p0, Tref, T0, cv,
           γ, pINF, c0, Γ0, s, e0,
           cs, τ1, μ, σY, n, PLASTIC,
           cα, τ2)

    return MP
