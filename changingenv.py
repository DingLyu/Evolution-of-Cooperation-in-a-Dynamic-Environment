import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np

T = 3000
lower = 0.5
higher = 1.5

env1 = [0.1]
env2 = [0.3]
env3 = [0.5]
env4 = [0.7]
env5 = [0.9]

for i in range(T):
    new_env1 = env1[-1] - 0.25 * env1[-1] * (1 - env1[-1]) * 0.01
    if new_env1 < 0:
        new_env1 = 0
        env1.append(0)
    elif new_env1 > 1:
        new_env1 = 1
        env1.append(1)
    else:
        env1.append(env1[-1] - 0.25 * env1[-1] * (1 - env1[-1]) * 0.01)

    new_env2 = env2[-1] - 0.25 * env2[-1] * (1 - env2[-1]) * 0.01
    if new_env2 < 0:
        new_env2 = 0
        env2.append(0)
    elif new_env2 > 1:
        new_env2 = 1
        env2.append(1)
    else:
        env2.append(env2[-1] - 0.25 * env2[-1] * (1 - env2[-1]) * 0.01)

    new_env3 = env3[-1] - 0.25 * env3[-1] * (1 - env3[-1]) * 0.01
    if new_env3 < 0:
        new_env3 = 0
        env3.append(0)
    elif new_env3 > 1:
        new_env3 = 1
        env3.append(1)
    else:
        env3.append(env3[-1] - 0.25 * env3[-1] * (1 - env3[-1]) * 0.01)

    new_env4 = env4[-1] - 0.25 * env4[-1] * (1 - env4[-1]) * 0.01
    if new_env4 < 0:
        new_env4 = 0
        env4.append(0)
    elif new_env4 > 1:
        new_env4 = 1
        env4.append(1)
    else:
        env4.append(env4[-1] - 0.25 * env4[-1] * (1 - env4[-1]) * 0.01)

    new_env5 = env5[-1] - 0.25 * env5[-1] * (1 - env5[-1]) * 0.01
    if new_env5 < 0:
        new_env5 = 0
        env5.append(0)
    elif new_env5 > 1:
        new_env5 = 1
        env5.append(1)
    else:
        env5.append(env5[-1] - 0.25 * env5[-1] * (1 - env5[-1]) * 0.01)


Y1 = []
Y2 = []
Y3 = []
Y4 = []
Y5 = []

for t in range(T+1):
    Y1.append(higher * env1[t] + lower * (1 - env1[t]))
    Y2.append(higher * env2[t] + lower * (1 - env2[t]))
    Y3.append(higher * env3[t] + lower * (1 - env3[t]))
    Y4.append(higher * env4[t] + lower * (1 - env4[t]))
    Y5.append(higher * env5[t] + lower * (1 - env5[t]))


Env1 = [0.1]
Env2 = [0.3]
Env3 = [0.5]
Env4 = [0.7]
Env5 = [0.9]

for i in range(T):
    new_env1 = Env1[-1] + 0.25 * Env1[-1] * (1 - Env1[-1]) * 0.01
    if new_env1 < 0:
        new_env1 = 0
        Env1.append(0)
    elif new_env1 > 1:
        new_env1 = 1
        Env1.append(1)
    else:
        Env1.append(Env1[-1] + 0.25 * Env1[-1] * (1 - Env1[-1]) * 0.01)

    new_env2 = Env2[-1] + 0.25 * Env2[-1] * (1 - Env2[-1]) * 0.01
    if new_env2 < 0:
        new_env2 = 0
        Env2.append(0)
    elif new_env2 > 1:
        new_env2 = 1
        Env2.append(1)
    else:
        Env2.append(Env2[-1] + 0.25 * Env2[-1] * (1 - Env2[-1]) * 0.01)

    new_env3 = Env3[-1] + 0.25 * Env3[-1] * (1 - Env3[-1]) * 0.01
    if new_env3 < 0:
        new_env3 = 0
        Env3.append(0)
    elif new_env3 > 1:
        new_env3 = 1
        Env3.append(1)
    else:
        Env3.append(Env3[-1] + 0.25 * Env3[-1] * (1 - Env3[-1]) * 0.01)

    new_env4 = Env4[-1] + 0.25 * Env4[-1] * (1 - Env4[-1]) * 0.01
    if new_env4 < 0:
        new_env4 = 0
        Env4.append(0)
    elif new_env4 > 1:
        new_env4 = 1
        Env4.append(1)
    else:
        Env4.append(Env4[-1] + 0.25 * Env4[-1] * (1 - Env4[-1]) * 0.01)

    new_env5 = Env5[-1] + 0.25 * Env5[-1] * (1 - Env5[-1]) * 0.01
    if new_env5 < 0:
        new_env5 = 0
        Env5.append(0)
    elif new_env5 > 1:
        new_env5 = 1
        Env5.append(1)
    else:
        Env5.append(Env5[-1] + 0.25 * Env5[-1] * (1 - Env5[-1]) * 0.01)


H1 = []
H2 = []
H3 = []
H4 = []
H5 = []

for t in range(T+1):
    H1.append(higher * Env1[t] + lower * (1 - Env1[t]))
    H2.append(higher * Env2[t] + lower * (1 - Env2[t]))
    H3.append(higher * Env3[t] + lower * (1 - Env3[t]))
    H4.append(higher * Env4[t] + lower * (1 - Env4[t]))
    H5.append(higher * Env5[t] + lower * (1 - Env5[t]))

plt.figure(figsize=(18, 6))

plt.subplot(121)
plt.plot(range(T+1), np.array(Env1), label='env1')
plt.plot(range(T+1), np.array(Env2), label='env2')
plt.plot(range(T+1), np.array(Env3), label='env3')
plt.plot(range(T+1), np.array(Env4), label='env4')
plt.plot(range(T+1), np.array(Env5), label='env5')
plt.plot(range(T+1), np.array(env1), linestyle='--', label='env6')
plt.plot(range(T+1), np.array(env2), linestyle='--', label='env7')
plt.plot(range(T+1), np.array(env3), linestyle='--', label='env8')
plt.plot(range(T+1), np.array(env4), linestyle='--', label='env9')
plt.plot(range(T+1), np.array(env5), linestyle='--', label='env10')
plt.xlabel("$t$", fontsize=25)
plt.xticks(fontsize=20)
plt.ylabel("$x_t$", fontsize=25)
plt.yticks(fontsize=20)
plt.legend(title="Dynamic Environment", title_fontsize=20, fontsize=15, frameon=False)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.subplot(122)
plt.plot(range(T+1), np.array(H1), label='env1')
plt.plot(range(T+1), np.array(H2), label='env2')
plt.plot(range(T+1), np.array(H3), label='env3')
plt.plot(range(T+1), np.array(H4), label='env4')
plt.plot(range(T+1), np.array(H5), label='env5')
plt.plot(range(T+1), np.array(Y1), linestyle='--', label='env6')
plt.plot(range(T+1), np.array(Y2), linestyle='--', label='env7')
plt.plot(range(T+1), np.array(Y3), linestyle='--', label='env8')
plt.plot(range(T+1), np.array(Y4), linestyle='--', label='env9')
plt.plot(range(T+1), np.array(Y5), linestyle='--', label='env10')
plt.xlabel("$t$", fontsize=25)
plt.xticks(fontsize=20)
plt.ylabel("$n_t$", fontsize=25)
plt.yticks([0.5, 1.5], ["$n_{\min}$", "$n_{\max}$"],fontsize=20)
plt.legend(title="Dynamic Environment", title_fontsize=20, fontsize=15, frameon=False)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.13)
plt.show()
