STIFFENED_GAS = 0
SHOCK_MG = 1


def eos_text_to_code(text):
    if text == 'sg':
        return STIFFENED_GAS
    if text == 'smg':
        return SHOCK_MG


def Γ_MG(ρ, MP):
    """ Returns the Mie-Gruneisen parameter
    """
    EOS = MP.EOS

    if EOS == STIFFENED_GAS:
        γ = MP.γ
        return γ - 1

    elif EOS == SHOCK_MG:
        Γ0 = MP.Γ0
        ρ0 = MP.ρ0
        return Γ0 * ρ0 / ρ


def p_ref(ρ, MP):
    """ Returns the reference pressure in the Mie-Gruneisen EOS
    """
    EOS = MP.EOS

    if EOS == STIFFENED_GAS:
        return -MP.pINF

    elif EOS == SHOCK_MG:
        c02 = MP.c02
        ρ0 = MP.ρ0
        s = MP.s
        if ρ > ρ0:
            return c02 * (1 / ρ0 - 1 / ρ) / (1 / ρ0 - s * (1 / ρ0 - 1 / ρ))**2
        else:
            return c02 * (ρ - ρ0)


def e_ref(ρ, MP):
    """ Returns the reference energy for the Mie-Gruneisen EOS
    """
    EOS = MP.EOS

    if EOS == STIFFENED_GAS:
        return MP.pINF / ρ

    elif EOS == SHOCK_MG:
        ρ0 = MP.ρ0
        pr = p_ref(ρ, MP)
        if ρ > ρ0:
            return 0.5 * pr * (1 / ρ0 - 1 / ρ)
        else:
            return 0


def dΓ_MG(ρ, MP):
    """ Returns the derivative of the Mie-Gruneisen parameter
    """
    EOS = MP.EOS

    if EOS == STIFFENED_GAS:
        return 0

    elif EOS == SHOCK_MG:
        Γ0 = MP.Γ0
        ρ0 = MP.ρ0
        return -Γ0 * ρ0 / ρ**2


def dp_ref(ρ, MP):
    """ Returns the derivative of the reference pressure in the Mie-Gruneisen EOS
    """
    EOS = MP.EOS

    if EOS == STIFFENED_GAS:
        return 0

    elif EOS == SHOCK_MG:
        c02 = MP.c02
        ρ0 = MP.ρ0
        s = MP.s
        if ρ > ρ0:
            return c02 * ρ0**2 * (s * (ρ0 - ρ) - ρ) / (s * (ρ - ρ0) - ρ)**3
        else:
            return c02


def de_ref(ρ, MP):
    """ Returns the derivative of the reference energy for the Mie-Gruneisen EOS
    """
    EOS = MP.EOS

    if EOS == STIFFENED_GAS:
        return -MP.pINF / ρ**2

    elif EOS == SHOCK_MG:
        c02 = MP.c02
        ρ0 = MP.ρ0
        s = MP.s
        if ρ > ρ0:
            return -(ρ - ρ0) * ρ0 * c02 / (s * (ρ - ρ0) - ρ)**3
        else:
            return 0


def internal_energy(ρ, p, MP):
    """ Returns the Mie-Gruneisen internal energy
    """
    Γ = Γ_MG(ρ, MP)
    pr = p_ref(ρ, MP)
    er = e_ref(ρ, MP)
    return er + (p - pr) / (ρ * Γ)


def pressure(ρ, e, MP):
    """ Returns the Mie-Gruneisen pressure, given the density and internal
        energy
    """
    Γ = Γ_MG(ρ, MP)
    pr = p_ref(ρ, MP)
    er = e_ref(ρ, MP)
    return (e - er) * ρ * Γ + pr


def temperature(ρ, p, MP):
    """ Returns the Mie-Gruneisen temperature, given the density and pressure
    """
    cv = MP.cv
    Tref = MP.Tref
    Γ = Γ_MG(ρ, MP)
    pr = p_ref(ρ, MP)
    return Tref + (p - pr) / (ρ * Γ * cv)


def dedρ(ρ, p, MP):
    """ Returns the derivative of the Mie-Gruneisen internal energy
        with respect to ρ
    """
    Γ = Γ_MG(ρ, MP)
    dΓ = dΓ_MG(ρ, MP)
    pr = p_ref(ρ, MP)
    dpr = dp_ref(ρ, MP)
    der = de_ref(ρ, MP)
    return der - (dpr * ρ * Γ + (Γ + ρ * dΓ) * (p - pr)) / (ρ * Γ)**2


def dedp(ρ, MP):
    """ Returns the derivative of the Mie-Gruneisen internal energy
        with respect to p
    """
    Γ = Γ_MG(ρ, MP)
    return 1 / (ρ * Γ)


def dTdρ(ρ, p, MP):
    """ Returns the derivative of the Mie-Gruneisen temperature
        with respect to ρ
    """
    cv = MP.cv
    Γ = Γ_MG(ρ, MP)
    dΓ = dΓ_MG(ρ, MP)
    pr = p_ref(ρ, MP)
    dpr = dp_ref(ρ, MP)
    return -(dpr * ρ * Γ + (Γ + ρ * dΓ) * (p - pr)) / (ρ * Γ)**2 / cv


def dTdp(ρ, MP):
    """ Returns the derivative of the Mie-Gruneisen temperature
        with respect to p
    """
    cv = MP.cv
    return dedp(ρ, MP) / cv
