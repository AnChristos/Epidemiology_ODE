from RK4 import RK4
import numpy as np


# Removal rate (beta) 1/3 day-1
# Infected people that cease to take part in the transmission per day
beta = 1. / 3.  # day^{-1}


# R0=3 , 
# leads to alpha = 1 day-1
# One S<->I contact lead to a new infection per day
def alpha(time, beta):
    R0 = 3.0
    return R0 * beta


def testSystem(time, vector_S_I_R):
    """
    dI/dt = alpha * I * S - beta*I
    dR/dt = beta * I
    dS/dt = -dI/dt-dR/dt
    """
    dI = alpha(time, beta) * vector_S_I_R[1] * \
        vector_S_I_R[0] - beta * vector_S_I_R[1]
    dR = beta * vector_S_I_R[1]
    dS = -1 * dI - 1 * dR
    return np.array([dS, dI, dR])


if __name__ == "__main__":
    # NSteps
    time = 80
    Step = 0.1
    NSteps = int(time / Step)

    initI = 1e-05  # Initial percentage of Infected
    initS = 1. - initI  # Initial percentage of Susceptible
    initR = 0  # Initial percentage of Recovered
    myrk4 = RK4(testSystem, Step, 0, np.array([initS, initI, initR]))

    t = np.arange(0.0, time, Step)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    for i in range(0, NSteps):
        S[i] = myrk4.currentValues()[0]
        I[i] = myrk4.currentValues()[1]
        R[i] = myrk4.currentValues()[2]
        myrk4.integrateStep()

    # plotting
    import matplotlib.pyplot as plt
    yticks = np.arange(0, 1.1, 1e-01)
    fig, ax = plt.subplots()
    ax.plot(t, S, label='S')
    ax.plot(t, I, label='I')
    ax.plot(t, R, label='R')
    ax.set_yticks(yticks)
    from matplotlib.ticker import FormatStrFormatter
    ax.set(xlabel='time (days)', ylabel='Fraction', title='SIR')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    # ax.set_yscale('log')
    legend = ax.legend(loc='center right')
    fig.savefig("SIR.png")

    fig, ax = plt.subplots()
    ax.plot(t, I, label='I')
    legend = ax.legend(loc='center right')
    fig.savefig("I.png")
