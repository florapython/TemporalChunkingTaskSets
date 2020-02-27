# Simplified code for the article "Temporal Chunking as a Mechanism for Unsupervised Learning of Task-Sets"
# Bouchacourt, Palminteri, Koechlin, Ostojic, 2020

# Used to simulate the RECURRENT SESSION, permits to reproduce figure 3 of the paper

from Generate_data import *
import matplotlib
from matplotlib import pyplot as plt
import numpy
import pdb


# ---------------------- PARAMETERS TO CHOOSE FROM ----------------------

inference = 1    # 0 if AN model, 1 if ANTN model (with TN inference)
nombre_param_an = 3 # number of independent parameters in the AN
add_trickytrial = True # whether to add tricky trials or not (in the behavioral paradigm, 10% of these are added)
random_ts = False # whether the order of task-sets is randomized or not

number_sessions_simulated = 500 # number of sessions simulated
number_episodes_simulated = 25  # number of episodes per session
number_trials_per_episode = 50 # number of trials per episode
param_antn = [0.4,0.4,0.4,0.4,0,0,7,0,0.17,0.017,0.5,0.7] # list of parameters :
# First four parameters in the AN are all equal in the paper : potentiation if rewarded (chosen association) ; potentiation if not rewarded (for other non chosen associations from the same stimulus) ; depression if rewarded (non chosen associations from the same stimulus) ; depression if not rewarded (for the chosen association, if incorrect)
# The two next parameters encode competition with respect to other stimuli than the presented one at each trial, it's set to zero in the paper (it did not improve fits in the Recurrent session)
# Then comes beta and epsilon for directed and undirected exploration
# Then the parameters of the TN : potentiation Qp and depression Qm
# Finally, the inhibition threshold gI, and the inference strength Jinc
#Total 12 parameters (5 parameters in the paper, cf Methods)



# ---------------------- SIMULATE THE RECURRENT SESSION ----------------------

MyGenerate = Generate_data(inference, number_sessions_simulated , number_episodes_simulated, number_trials_per_episode, param_antn, add_trickytrial, random_ts) 
Stimuli, Correct_actions, Chosen_actions_from_model, Tricky_trials, Rewards, Task_sets, Matrix_connexions_TN_TS1, Matrix_connexions_TN_TS2,Matrix_connexions_TN_TS3, Spurious_connexions, Matrix_inference_fromTNtoAN = MyGenerate.generate_fictive_data()

# Average over sessions
Performance = numpy.mean(Rewards,axis=0)
Inference_strength = numpy.mean(Matrix_inference_fromTNtoAN,axis=0)
Connexions_TN_TS1 = numpy.mean(Matrix_connexions_TN_TS1,axis=0)
Connexions_TN_TS2 = numpy.mean(Matrix_connexions_TN_TS2,axis=0)
Connexions_TN_TS3 = numpy.mean(Matrix_connexions_TN_TS3,axis=0)
Spurious_connexions = numpy.mean(Spurious_connexions,axis=0)
fontsize_ylabel = 10
linewidth_graph = 6

# ---------------------- PLOT RESULTS (FIGURE 3 OF THE PAPER) ----------------------
ziou=matplotlib.figure.SubplotParams(left=0.1,bottom=0.1,right=0.95,top=0.95,hspace=0.45,wspace=0.2)
fig = plt.figure(figsize=(10,10),subplotpars=ziou)

ax1 = plt.subplot(3,1,1)
plt.ylabel('TN synaptic weights', fontsize=fontsize_ylabel)
plt.plot(Connexions_TN_TS1.flatten(), 'b-', linewidth=linewidth_graph,label = 'Task set 1')
plt.plot(Connexions_TN_TS2.flatten(), 'c-', linewidth=linewidth_graph,label = 'Task set 2')
plt.plot(Connexions_TN_TS3.flatten(), 'g-', linewidth=linewidth_graph,label = 'Task set 3')
plt.plot(Spurious_connexions.flatten(), 'y-', linewidth=linewidth_graph,label = 'Spurious')
plt.legend()

ax2 = plt.subplot(3,1,2)
plt.ylabel('TN inference strength to AN', fontsize=fontsize_ylabel)
plt.plot(Inference_strength.flatten(), 'k-', linewidth=linewidth_graph)

ax3 = plt.subplot(3,1,3)
plt.ylabel('Proportion correct', fontsize=fontsize_ylabel)
plt.plot(Performance.flatten(), 'k-', linewidth=linewidth_graph)

plt.show()



