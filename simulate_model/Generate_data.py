from Model import *
import numpy
import pdb


class Generate_data:
  def __init__(self, inference, number_sessions_simulated , number_episodes_simulated, number_trials_per_episode, param_antn, add_trickytrials, random_ts) :
    self.inference = inference
    self.number_sessions_simulated = number_sessions_simulated 
    self.number_episodes_simulated = number_episodes_simulated
    self.number_trials_per_episode = number_trials_per_episode
    self.param_antn = param_antn # list of parameters
    self.add_trickytrials = add_trickytrials # equals True of False, if simulation includes noisy trials
    self.random_ts = random_ts # whether task-sets are ordered throughout episodes or not, ordered makes visualisation easier (cf Figure 3 of the paper)


  def generate_tt(self) : # Generates the tricky trials (10% noise in the experimental design, cf Methods)
    Matrix_of_tricky_trials = numpy.zeros(self.number_trials_per_episode,dtype=numpy.int64)
    if self.add_trickytrials :
      for index in xrange(self.number_trials_per_episode) :
        p = numpy.random.rand()
        Matrix_of_tricky_trials[index] = (0,1)[p<=0.1]
    return Matrix_of_tricky_trials

  def generate_stim(self) : # Generates the list of stimuli
    List = numpy.zeros(self.number_trials_per_episode,dtype=numpy.int64)
    for index in xrange(self.number_trials_per_episode) :
      random_number  = numpy.random.rand()
      if random_number< 1./3 :
        List[index] = 1
      elif random_number < 2./3 :
        List[index] = 2
      else :
        List[index] = 3
    return List

  def generate_task_sets(self): # Generates a list of task-sets to be presented to the model
    Task_set = numpy.zeros((self.number_episodes_simulated,4),dtype=numpy.int64) 
    Numero_TS = numpy.zeros(self.number_episodes_simulated,dtype=numpy.int64)
    # task-sets in the recurrent session
    Stim_TS = numpy.zeros((4,4),dtype=numpy.int64) # ts, stim
    Stim_TS[1,1]=1
    Stim_TS[1,2]=2
    Stim_TS[1,3]=3
    Stim_TS[2,1]=2
    Stim_TS[2,2]=3
    Stim_TS[2,3]=4
    Stim_TS[3,1]=3
    Stim_TS[3,2]=4 
    Stim_TS[3,3]=1
    for index_episode in xrange(self.number_episodes_simulated) :
      if self.random_ts : # if the order of task-sets is randomized, note than a task-set can't be repeated in the successive episode, per experimental design
        if index_episode == 0 :
          p = numpy.random.rand()
          if p < 1./3:
            numero_ts = 1
          elif p < 2./3 :
            numero_ts = 2
          else :
            numero_ts = 3
        else :
          p = numpy.random.rand()
          if p < 0.5 :
            numero_ts = Numero_TS[index_episode-1] +1
            if numero_ts == 4 :
              numero_ts = 1
          else :
            numero_ts = Numero_TS[index_episode-1] -1
            if numero_ts == 0 :
              numero_ts = 3
      else : # if task-sets are ordered for visualisation purposes
        numero_ts = index_episode%3  + 1  
      Numero_TS[index_episode] = numero_ts
      Task_set[index_episode,1:]= Stim_TS[numero_ts,1:] 
    return Task_set, Numero_TS


  def generate_fictive_expe(self) : # Generates a fictive experimental design
    Stim_simu = numpy.zeros((self.number_sessions_simulated, self.number_episodes_simulated, self.number_trials_per_episode),dtype=numpy.int64)
    Correct_action_simu = numpy.zeros((self.number_sessions_simulated, self.number_episodes_simulated, self.number_trials_per_episode),dtype=numpy.int64)
    Task_sets_simu = numpy.zeros((self.number_sessions_simulated, self.number_episodes_simulated, 4),dtype=numpy.int64)
    Tricky_trials_simu = numpy.zeros((self.number_sessions_simulated, self.number_episodes_simulated, self.number_trials_per_episode),dtype=numpy.int64)
    Numero_TS_simu = numpy.zeros((self.number_sessions_simulated, self.number_episodes_simulated),dtype=numpy.int64)
    for index_session in xrange(self.number_sessions_simulated) :
      Task_set_simulated, Numero_TS = self.generate_task_sets()
      Task_sets_simu[index_session,:,:] = Task_set_simulated
      Numero_TS_simu[index_session,:] = Numero_TS
      for index_episode in xrange(self.number_episodes_simulated) :
        Tricky_trials_simu[index_session, index_episode,:] = self.generate_tt()
        Stim_simu[index_session, index_episode,:] = self.generate_stim()
        for index_trial in xrange(self.number_trials_per_episode):
          Correct_action_simu[index_session, index_episode,index_trial]  = Task_sets_simu[index_session, index_episode, Stim_simu[index_session, index_episode, index_trial]]
    return Stim_simu,Correct_action_simu,Tricky_trials_simu, Task_sets_simu, Numero_TS_simu



  def connexions_per_ts(self, Matrix_connexions_TN) : # Study the connexions in the TN per task-set
    Matrix_connexions_TN_TS1 = numpy.zeros((self.number_sessions_simulated,self.number_episodes_simulated,self.number_trials_per_episode))
    Matrix_connexions_TN_TS2 = numpy.zeros((self.number_sessions_simulated,self.number_episodes_simulated,self.number_trials_per_episode))
    Matrix_connexions_TN_TS3 = numpy.zeros((self.number_sessions_simulated,self.number_episodes_simulated,self.number_trials_per_episode))
    Spurious_connexions = numpy.zeros((self.number_sessions_simulated,self.number_episodes_simulated,self.number_trials_per_episode))

    List_co_ts1 = [0,1,2]
    List_co_ts2 = [4,5,6]
    List_co_ts3 = [8,9,10]
    for index_ts1 in List_co_ts1 :
      for index_ts1_bis in List_co_ts1 :
        if index_ts1_bis!=index_ts1 :
          Matrix_connexions_TN_TS1+=Matrix_connexions_TN[:,:,:,index_ts1,index_ts1_bis]
    for index_ts2 in List_co_ts2 :
      for index_ts2_bis in List_co_ts2 :
        if index_ts2_bis!=index_ts2 :
          Matrix_connexions_TN_TS2+=Matrix_connexions_TN[:,:,:,index_ts2,index_ts2_bis]
    for index_ts3 in List_co_ts3 :
      for index_ts3_bis in List_co_ts3 :
        if index_ts3_bis!=index_ts3 :
          Matrix_connexions_TN_TS3+=Matrix_connexions_TN[:,:,:,index_ts3,index_ts3_bis]
    Spurious_connexions = numpy.sum(numpy.sum(Matrix_connexions_TN,axis=4),axis=3)-Matrix_connexions_TN_TS1-Matrix_connexions_TN_TS2-Matrix_connexions_TN_TS3
    Matrix_connexions_TN_TS1/=6
    Matrix_connexions_TN_TS2/=6
    Matrix_connexions_TN_TS3/=6
    Spurious_connexions/=114

    return Matrix_connexions_TN_TS1, Matrix_connexions_TN_TS2,Matrix_connexions_TN_TS3, Spurious_connexions



  def generate_fictive_data(self) : # Generates data and simulate the model
    Stim_simu, Correct_action_simu, Tricky_trials_simu, Task_sets_simu, Numero_TS_simu = self.generate_fictive_expe()
    Chosen_action_model = numpy.zeros((self.number_sessions_simulated,self.number_episodes_simulated,self.number_trials_per_episode),dtype=numpy.int64)
    Reward_simu = numpy.zeros((self.number_sessions_simulated,self.number_episodes_simulated,self.number_trials_per_episode),dtype=numpy.int64)
    Matrix_connexions_TN = numpy.zeros((self.number_sessions_simulated,self.number_episodes_simulated,self.number_trials_per_episode,12,12))
    Matrix_inference_fromTNtoAN = numpy.zeros((self.number_sessions_simulated,self.number_episodes_simulated,self.number_trials_per_episode,4,5))

    # Parameters description
    param_an = self.param_antn[:8] # 6 parameters for potentiation/depression in the AN, plus beta and epsilon (directed and undirected exploration)
    param_tn = [self.param_antn[8],self.param_antn[9]] # potentiation and depression in the TN
    gI = self.param_antn[10] # inhibition
    Jinc = self.param_antn[11] # inference strength
    for index_session in xrange(self.number_sessions_simulated) :
      MyModel = Model(Stim_simu[index_session], Task_sets_simu[index_session], Tricky_trials_simu[index_session], self.number_episodes_simulated, self.number_trials_per_episode, param_an, param_tn, gI, Jinc, self.inference)
      Action_model, Reward_model, J_MAT, Inference_matrix = MyModel.run_free_simulation()
      Chosen_action_model[index_session] = Action_model
      Reward_simu[index_session] = Reward_model
      Matrix_connexions_TN[index_session] = J_MAT
      Matrix_inference_fromTNtoAN[index_session] = Inference_matrix

    Matrix_connexions_TN_TS1, Matrix_connexions_TN_TS2,Matrix_connexions_TN_TS3, Spurious_connexions = self.connexions_per_ts(Matrix_connexions_TN)
    Matrix_inference_fromTNtoAN = numpy.mean(numpy.mean(Matrix_inference_fromTNtoAN,axis=4),axis=3)


    return Stim_simu, Correct_action_simu, Chosen_action_model, Tricky_trials_simu, Reward_simu, Task_sets_simu, Matrix_connexions_TN_TS1, Matrix_connexions_TN_TS2,Matrix_connexions_TN_TS3, Spurious_connexions, Matrix_inference_fromTNtoAN
