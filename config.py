
class ConfigPP:
    T = 20.0
    T_unit = 0.001
    N = int(T / T_unit)
    U_start = 30.0
    V_start = 3.33

    alpha = 1.0
    beta = 3.0
    gamma = 0.3
    e = 0.333

    ub = T
    lb = 0.0

    y_ub = 100.0
    y_lb = 0.0


class ConfigSIS:
    T = 100.0
    T_unit = 0.01
    N = int(T / T_unit)
    S_start = 99.0
    I_start = 1.0
    R_start = 0.0
    SIR_sum = 1000.0

    beta = 0.01
    gamma = 0.05

    ub = T
    lb = 0.0

