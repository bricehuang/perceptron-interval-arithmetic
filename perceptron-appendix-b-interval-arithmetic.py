import flint as fl

#########################################################################################
# This file implements computer-assisted interval arithmetic proofs, used in Appendix B #
# of the paper "Capacity threshold for the Ising perceptron" by B. Huang.               #
#                                                                                       #
# The interval arithmetic is implemented in Python 3 using the package python-flint,    #
# which wraps the interval arithmetic package Arb in C++.                               #
#                                                                                       #
# The file is structured as follows. We first verify Claims B.1 through B.4 from the    #
# paper (Subsection B.4). These are the most computationally intensive checks,          #
# requiring rigorous numerical bounds on certain gaussian integrals.                    #
#                                                                                       #
# After this, we verify the remaining numerical inequalities used in Appendix B. These  #
# are much simpler, amounting to comparing arithmetic expressions of explicitly defined #
# decimal numbers.                                                                      #
#########################################################################################

#######################
# Top level constants #
#######################

ALPHA_LB = fl.acb(0.833078599)
ALPHA_UB = fl.acb(0.833078600)
Q_LB = fl.acb(0.56394907949)
Q_UB = fl.acb(0.56394908030)
PSI_LB = fl.acb(2.5763513100)
PSI_UB = fl.acb(2.5763513224)
GAMMA_UB = Q_UB / (1 - Q_UB)
GAMMA_LB = Q_LB / (1 - Q_LB)
Z_HAT = fl.acb(-0.669316)

def gaussian_density(x):
    return (-x**2/2).exp() / (2*fl.arb.pi()).sqrt()

#############################
# Verification of Claim B.1 #
#############################

P4_UB = fl.acb(0.4405902320)
P4_LB = fl.acb(0.4405902310)

def th(x):
    return ((2*x).exp() - 1) / ((2*x).exp() + 1)

def p4_integrand(x,psi):
    return (th(x * psi**.5) ** 4) * gaussian_density(x)

def verify_claim_b1():
    print("Verifying Claim B.1")
    L = fl.acb(10)

    # for the upper bound, numerically integrate on [-L,+L]
    # and bound contribution from outside this interval
    p4_ub_middle = fl.acb.integral(lambda x,_: p4_integrand(x,PSI_UB),-L,+L)
    p4_ub_ends = 2 * (-L**2/2).exp()
    p4_ub_calculated = p4_ub_middle + p4_ub_ends
    print("should be positive: %s" % (P4_UB - p4_ub_calculated))

    # for the lower bound, numerically integrate on [-L,+L]
    # and discard contribution from outside this interval
    p4_lb_calculated = fl.acb.integral(lambda x,_: p4_integrand(x,PSI_LB),-L,+L)
    print("should be positive: %s" % (p4_lb_calculated - P4_LB))

verify_claim_b1()

#############################
# Verification of Claim B.3 #
#############################

def ch(x):
    return (x.exp() + (-x).exp()) / 2

def m_integrand(x,z,psi):
    return (z + ch(x * psi**.5) ** 2) ** -1 * gaussian_density(x)

M_UB = 0.9309695
def verify_claim_b3():
    print("Verifying Claim B.3")

    # numerically integrate on [-L,+L] and bound contribution from outside this interval
    L = fl.acb(10)
    m_ub_middle = fl.acb.integral(lambda x,_: m_integrand(x,Z_HAT,PSI_LB),-L,+L)
    m_ub_ends = 2 * (-L**2/2).exp() / (1 + Z_HAT)
    m_ub_calculated = m_ub_middle + m_ub_ends
    print("should be positive: %s" % (M_UB - m_ub_calculated))

verify_claim_b3()

#############################
# Verification of Claim B.2 #
#############################

R4_UB = fl.acb(5.317)
R4_LB = fl.acb(5.297)

def cE(x):
    # Evaluates cE(x) = psi(x) / Psi(x), with small (quantifiable) multiplicative error for x <= 10.
    Lplus = fl.acb(12)
    E_inv_main_integral = fl.acb.integral(lambda y,_: (- (y**2-x**2) / 2).exp(), x, Lplus)
    E_inv_error_term = fl.acb(fl.arb(0,(- (Lplus**2-x**2) / 2).exp().abs_upper()))
    E_inv = E_inv_main_integral + E_inv_error_term
    return E_inv ** -1

def verify_claim_b2():
    print("Verifying Claim B.2")
    L = fl.acb(8)
    delta = fl.acb(0.001)
    J = 8000

    # upper bound
    def r4_ub_term(j):
        xj = j * delta
        xj_plus_one = (j+1) * delta
        cE_4th_power = cE(xj_plus_one * GAMMA_UB ** .5) ** 4
        if j<0:
            return gaussian_density(xj_plus_one) * cE_4th_power
        else:
            return gaussian_density(xj) * cE_4th_power
    r4_ub_terms = [r4_ub_term(j) for j in range(-J,J)]
    r4_ub_error_term = 16 * (1 + 11 * GAMMA_UB) * (-L**2/4).exp()
    r4_ub_calculated = delta * sum(r4_ub_terms) + r4_ub_error_term
    print("should be positive: %s" % (R4_UB - r4_ub_calculated))

    # lower bound
    def r4_lb_term(j):
        xj = j * delta
        xj_plus_one = (j+1) * delta
        cE_4th_power = cE(xj * GAMMA_LB ** .5) ** 4
        if j<0:
            return gaussian_density(xj) * cE_4th_power
        else:
            return gaussian_density(xj_plus_one) * cE_4th_power
    r4_lb_terms = [r4_lb_term(j) for j in range(-J,J)]
    r4_lb_calculated = delta * sum(r4_lb_terms)
    print("should be positive: %s" % (r4_lb_calculated - R4_LB))

verify_claim_b2()

#############################
# Verification of Claim B.4 #
#############################

G_LB = 0.7739
def hat_g(x):
    cE_eval = cE(x)
    cE_prime_eval = cE_eval * (cE_eval - x)
    return cE_prime_eval / ((1 - Q_LB) * (1 - cE_prime_eval) + M_UB * cE_prime_eval)

def verify_claim_b4():
    print("Verifying Claim B.4")
    delta = fl.acb(0.001)
    J = 8000
    def g_lb_term(j):
        xj = j * delta
        xj_plus_one = (j+1) * delta
        hat_g_eval = hat_g(xj * GAMMA_LB ** .5)
        if j<0:
            return gaussian_density(xj) * hat_g_eval
        else:
            return gaussian_density(xj_plus_one) * hat_g_eval
    g_lb_terms = [g_lb_term(j) for j in range(-J,J)]
    g_lb_calculated = delta * sum(g_lb_terms) - 20 * (GAMMA_UB - GAMMA_LB)
    print("should be positive: %s" % (g_lb_calculated - G_LB))

verify_claim_b4()

#############################
# Verification of Claim B.7 #
#############################

A_UB = fl.acb(0.5446)
def verify_claim_b7():
    print("Verifying Claim B.7")
    term1 = 1 - 2 * Q_LB + P4_UB
    term2 = GAMMA_UB * PSI_UB / (1 + Q_LB) + ALPHA_LB * (1 - GAMMA_LB) / ((1 + Q_LB) * (1 + 2 * Q_LB)) * R4_LB
    a_ub_calculated = term1 * term2
    print("should be positive: %s" % (A_UB - a_ub_calculated))

verify_claim_b7()

#############################
# Verification of Claim B.8 #
#############################

LAMBDA_UB = -0.1906
def verify_claim_b8():
    print("Verifying Claim B.8")
    lambda_ub_calculated = Z_HAT - ALPHA_LB * G_LB + (1 - Q_LB) * PSI_UB
    print("should be positive: %s" % (LAMBDA_UB - lambda_ub_calculated))

verify_claim_b8()

##############################
# Verification of Claim B.10 #
##############################

C1_LB = fl.acb(-0.7193)
C1_UB = fl.acb(-0.7165)
C2_LB = fl.acb( 5.0439)
C2_UB = fl.acb( 5.0568)
C3_LB = fl.acb( 1.1345)
C3_UB = fl.acb( 1.1526)
def verify_claim_b10():
    print("Verifying Claim B.10")

    C1_lb_calculated = - ALPHA_UB * R4_UB / (PSI_LB ** 2 * (1 - Q_UB) * (1 + 2*Q_LB))
    print("should be positive: %s" % (C1_lb_calculated - C1_LB))

    C1_ub_calculated = - ALPHA_LB * R4_LB / (PSI_UB ** 2 * (1 - Q_LB) * (1 + 2*Q_UB))
    print("should be positive: %s" % (C1_UB - C1_ub_calculated))

    C2_lb_calculated = 2 * (2-Q_UB) * ALPHA_LB * R4_LB / (PSI_UB * (1 - Q_LB**2) * (1+2*Q_UB)) + 2 * Q_LB / (1 - Q_LB**2)
    print("should be positive: %s" % (C2_lb_calculated - C2_LB))

    C2_ub_calculated = 2 * (2-Q_LB) * ALPHA_UB * R4_UB / (PSI_LB * (1 - Q_UB**2) * (1+2*Q_LB)) + 2 * Q_UB / (1 - Q_UB**2)
    print("should be positive: %s" % (C2_UB - C2_ub_calculated))

    C3_lb_calculated = - ALPHA_UB * R4_UB / ((1 - Q_UB) * (1 + 2*Q_LB)) + PSI_LB / (1 - Q_LB)
    print("should be positive: %s" % (C3_lb_calculated - C3_LB))

    C3_ub_calculated = - ALPHA_LB * R4_LB / ((1 - Q_LB) * (1 + 2*Q_UB)) + PSI_UB / (1 - Q_UB)
    print("should be positive: %s" % (C3_UB - C3_ub_calculated))

verify_claim_b10()

##############################
# Verification of Claim B.11 #
##############################

I1_LB = fl.acb(0.24759912)
I1_UB = fl.acb(0.24759923)
I2_LB = fl.acb(0.16997315)
I2_UB = fl.acb(0.16997318)
I3_LB = fl.acb(0.12335884)
I3_UB = fl.acb(0.12335885)
def verify_claim_b11():
    print("Verifying Claim B.11")

    I1_lb_calculated = PSI_LB * (1 - Q_UB) - 2 * PSI_UB**2 * (1 - 4*Q_LB + 3*P4_UB)
    print("should be positive: %s" % (I1_lb_calculated - I1_LB))

    I1_ub_calculated = PSI_UB * (1 - Q_LB) - 2 * PSI_LB**2 * (1 - 4*Q_UB + 3*P4_LB)
    print("should be positive: %s" % (I1_UB - I1_ub_calculated))

    I2_lb_calculated = PSI_LB * (1 - 4*Q_UB + 3*P4_LB)
    print("should be positive: %s" % (I2_lb_calculated - I2_LB))

    I2_ub_calculated = PSI_UB * (1 - 4*Q_LB + 3*P4_UB)
    print("should be positive: %s" % (I2_UB - I2_ub_calculated))

    I3_lb_calculated = Q_LB - P4_UB
    print("should be positive: %s" % (I3_lb_calculated - I3_LB))

    I3_ub_calculated = Q_UB - P4_LB
    print("should be positive: %s" % (I3_UB - I3_ub_calculated))

verify_claim_b11()

##############################
# Verification of Claim B.12 #
##############################

M11_UB  = fl.acb(-0.045408)
M22_UB  = fl.acb(-0.020490)
M12_UB  = fl.acb(-0.025685)
M12_LB  = fl.acb(-0.026567)
Mdet_LB = fl.acb(0.0002246)
def verify_claim_b12():
    print("Verifying Claim B.12")

    M11_ub_calculated = - I1_LB + C1_UB * I1_LB ** 2 + C2_UB * I1_UB * I2_UB + C3_UB * I2_UB ** 2
    print("should be positive: %s" % (M11_UB - M11_ub_calculated))

    M22_ub_calculated = - I3_LB + C1_UB * I2_LB ** 2 + C2_UB * I2_UB * I3_UB + C3_UB * I3_UB ** 2
    print("should be positive: %s" % (M22_UB - M22_ub_calculated))

    M12_ub_calculated = - I2_LB + C1_UB * I1_LB * I2_LB + C2_UB * (I2_UB ** 2 + I1_UB * I3_UB) / 2 + C3_UB * I2_UB * I3_UB
    print("should be positive: %s" % (M12_UB - M12_ub_calculated))

    M12_lb_calculated = - I2_UB + C1_LB * I1_UB * I2_UB + C2_LB * (I2_LB ** 2 + I1_LB * I3_LB) / 2 + C3_LB * I2_LB * I3_LB
    print("should be positive: %s" % (M12_lb_calculated - M12_LB))

    Mdet_lb_calculated = M11_UB * M22_UB - M12_LB ** 2
    print("should be positive: %s" % (Mdet_lb_calculated - Mdet_LB))

verify_claim_b12()
