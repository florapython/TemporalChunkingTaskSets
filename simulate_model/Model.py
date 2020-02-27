import numpy


class Model: # Model one session
  def __init__(self, List_Stim, List_TaskSets, List_TrickyTrial, number_of_episodes, number_of_trials_per_episodes, param_an, param_tn, gI, Jinc, inference):
    self.List_Stim = List_Stim 
    self.List_TaskSets = List_TaskSets
    self.List_TrickyTrial = List_TrickyTrial
    self.number_of_episodes = number_of_episodes
    self.number_of_trials_per_episodes = number_of_trials_per_episodes
    self.param_an = param_an
    self.gI = gI
    self.Jinc = Jinc
    self.Qplus = param_tn[0]
    self.Qminus = param_tn[1]
    self.inference = inference


  def run_free_simulation(self) : # Runs the model as a free simulation
    Action_model = numpy.zeros((self.number_of_episodes,self.number_of_trials_per_episodes),dtype=numpy.int64)
    Reward_model = numpy.zeros((self.number_of_episodes,self.number_of_trials_per_episodes),dtype=numpy.int64)
    J_MAT_all=numpy.zeros((self.number_of_episodes,self.number_of_trials_per_episodes,12,12))
    Inference_matrix = numpy.zeros((self.number_of_episodes,self.number_of_trials_per_episodes,4,5))

    JMatrix = numpy.zeros((12,12)) 
    Cbefore = numpy.ones((4,5))*numpy.nan # stim, action
    Cafter = numpy.ones((4,5))*numpy.nan # stim, action
    Cafter[1:,1:]=0.25 # initialisation

    for index_episode in xrange(self.number_of_episodes) :
      for index_trial in xrange(self.number_of_trials_per_episodes) :
        # AN learning and decision making
        self.run_an(Cbefore, Cafter, JMatrix, Action_model, Reward_model, Inference_matrix, index_trial, index_episode)
        # TN learning
        if index_trial==0 and self.inference == 1 :
          stim_courant =  self.List_Stim[index_episode,index_trial]
          action_courante = Action_model[index_episode,index_trial]
          state_courant = self.correspondance(stim_courant,action_courante)
          JMatrix[:,:] = self.depression(state_courant,JMatrix[:,:])
          if index_episode > 0 :
            stim_avant = index_stim_ep_precedent
            action_avant = index_action_ep_precedent
            state_avant = self.correspondance(stim_avant,action_avant)
            JMatrix[:,:] = self.potentiation(state_avant,state_courant,JMatrix[:,:])
        elif index_trial > 0  and self.inference == 1 :
          stim_avant = self.List_Stim[index_episode,index_trial-1]
          action_avant = Action_model[index_episode,index_trial-1]
          state_avant = self.correspondance(stim_avant,action_avant)
          stim_courant =  self.List_Stim[index_episode,index_trial]
          action_courante = Action_model[index_episode,index_trial]
          state_courant = self.correspondance(stim_courant,action_courante)
          JMatrix[:,:] = self.potentiation(state_avant,state_courant,JMatrix[:,:])
          JMatrix[:,:] = self.depression(state_courant,JMatrix[:,:])
        J_MAT_all[index_episode,index_trial,:,:]=JMatrix[:,:]
      index_stim_ep_precedent = self.List_Stim[index_episode,self.number_of_trials_per_episodes-1]
      index_action_ep_precedent = Action_model[index_episode,self.number_of_trials_per_episodes-1]
    return Action_model, Reward_model, J_MAT_all, Inference_matrix

  def correspondance(self,s,a) : # a function to fill the Matrix of connexions in the TN
    s = int(round(s))
    a = int(round(a))
    state_number = int(round((s-1)*4+a-1))
    val = self.associer_indice_matrice(state_number)
    return val

  def correspondance_inverse(self,indice) :
    indice = int(round(indice))
    index = -1
    if indice == 0 :
      index = 0
    elif indice==4 :
      index = 1
    elif indice== 8 :
      index = 2
    elif indice == 3 :
      index = 3
    elif indice == 7:
      index = 4
    elif indice == 1 :
      index = 5
    elif indice == 5:
      index = 6
    elif indice == 9 :
      index = 7
    elif indice == 10 :
      index = 8
    elif indice == 11 :
      index = 9
    elif indice == 2 :
      index = 10
    elif indice == 6 :
      index = 11
    else :
      pdb.set_trace()
    action = index%4+1
    stim = index//4+1
    return stim, action


  def associer_indice_matrice(self,indice):
    if indice==0 :
      return 0
    elif indice==1 :
      return 4
    elif indice== 2 :
      return 8
    elif indice == 3 :
      return 3
    elif indice==4:
      return 7
    elif indice==5 :
      return 1
    elif indice == 6:
      return 5
    elif indice == 7 :
      return 9
    elif indice==8 :
      return 10
    elif indice == 9 :
      return 11
    elif indice == 10 :
      return 2
    elif indice == 11 :
      return 6


  

  def potentiation(self,i,j,Mat) : # potentiation in the TN
    Mat2 = numpy.zeros((12,12))
    Mat2[:,:] = Mat[:,:]
    # potentiation
    if i!=j :
      Mat2[i,j] += self.Qplus*(1.0-Mat[i,j]) 
    return Mat2

  def depression(self,i,Mat) : # depression in the TN
    Mat2 = numpy.zeros((12,12))
    Mat2[:,:] = Mat[:,:]
    # PRE activated depression
    for index_a in xrange(12) :
      if index_a!=i : 
        Mat2[i,index_a] -= self.Qminus*Mat[i,index_a]
    return Mat2




  def run_an(self, Cbefore, Cafter, JMat, Action_model, Reward_model, Inference_matrix, index_trial, index_episode): # Runs the AN, decision and learning
    QPR = min(1.,max(0.,self.param_an[0]))
    QPNR = min(1.,max(0.,self.param_an[1]))
    QMR = min(1.,max(0.,self.param_an[2]))
    QMNR = min(1.,max(0.,self.param_an[3]))
    QCOMP_P = min(1.,max(0.,self.param_an[4]))
    QCOMP_M = min(1.,max(0.,self.param_an[5]))
    beta = self.param_an[6]
    epsilon = self.param_an[7]

    # DECISION
    Cbefore[1:,1:] = Cafter[1:,1:]
    cur_s = self.List_Stim[index_episode,index_trial]
    proba_m = beta*Cbefore[cur_s,1:] 
    proba_m = numpy.exp(proba_m-numpy.amax(proba_m))
    proba_m=float(epsilon)/4 + (1.0-epsilon)*proba_m/numpy.sum(proba_m) # softmax
    correct_action = self.List_TaskSets[index_episode,cur_s]
    p = numpy.random.rand()
    if p <= proba_m[0]:
      model_action = 1
    elif p > proba_m[0] and p <= proba_m[0]+proba_m[1] :
      model_action = 2
    elif p > proba_m[0]+proba_m[1] and p <= proba_m[0]+proba_m[1]+proba_m[2] :
      model_action = 3
    elif p < proba_m[0]+proba_m[1]+proba_m[2]+proba_m[3] :
      model_action = 4
    if self.List_TrickyTrial[index_episode,index_trial]==1 :
      Reward_model[index_episode,index_trial] = (0,1)[model_action!=correct_action]
    elif self.List_TrickyTrial[index_episode,index_trial]==0 :
      Reward_model[index_episode,index_trial] = (0,1)[model_action==correct_action]
    Action_model[index_episode,index_trial] = model_action

    # UPDATE RULE IN AN AND INFERENCE FROM TN TO AN WHEN REWARD IS 1
    if Reward_model[index_episode,index_trial]==1 :
      Cafter[cur_s,model_action] = float(Cbefore[cur_s,model_action] + QPR*(1.0-Cbefore[cur_s,model_action]))
      for k in xrange(1,5,1) :
        if k!=model_action :
          Cafter[cur_s,k] = float(Cbefore[cur_s,k] - QMR*Cbefore[cur_s,k])   
      for index_s in xrange(1,4,1) :
        if index_s!=cur_s :
          Cafter[index_s,model_action] = float(Cbefore[index_s,model_action] - QCOMP_M*Cbefore[index_s,model_action])

      if self.inference==1 :
        evt_cur = int(round(self.correspondance(cur_s,model_action)))
        activate_itself=int(0)
        for index in xrange(12) :
          if index!= evt_cur :
            if JMat[evt_cur,index] >= self.gI :
              activate_itself = 1
              stim_co, action_co = self.correspondance_inverse(index)
              stim_co = int(round(stim_co))
              action_co = int(round(action_co))
              Cafter[stim_co,action_co] += float(self.Jinc*(1-Cafter[stim_co,action_co]))
              Inference_matrix[index_episode,index_trial,stim_co,action_co]=float(self.Jinc*(1-Cafter[stim_co,action_co]))
        if activate_itself==1 :
          Cafter[cur_s,model_action] += float(self.Jinc*(1-Cafter[cur_s,model_action]))
          Inference_matrix[index_episode,index_trial,cur_s,model_action] = float(self.Jinc*(1-Cafter[cur_s,model_action]))

    # UPDATE RULE IN AN WHEN REWARD IS 0
    elif Reward_model[index_episode,index_trial]==0 :
      Cafter[cur_s,model_action] = float(Cbefore[cur_s,model_action] - QMNR*Cbefore[cur_s,model_action])  
      for k in xrange(1,5,1) :
        if k!=model_action :
          Cafter[cur_s,k] = float(Cbefore[cur_s,k] + QPNR*(1-Cbefore[cur_s,k]))
      for index_s in xrange(1,4,1) :
        if index_s!=cur_s :
          Cafter[index_s,model_action] = float(Cbefore[index_s,model_action] + QCOMP_P*(1-Cbefore[index_s,model_action]))

    return





