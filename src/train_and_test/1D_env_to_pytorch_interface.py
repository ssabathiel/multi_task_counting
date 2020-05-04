###############
## Prepare data set
##################
import torch


print("Load env-to-pytorch interface .... ")

### Higher level task
def create_task_data():
  env = CountEnv()

  epoch_duration = 200

  height = env.observation[:,0].size
  width = env.observation[0,:].size

  all_imgs = np.zeros([epoch_duration,height, width])
  all_actions = np.zeros([epoch_duration,env.n_actions])
    
  for t in range(epoch_duration):
    env.update("left")                       ## !!!!!!!! WHY is this here
    all_imgs[t,:,:] = env.observation/255
    all_actions[t,:] = env.action_onehot
  
  all_imgs = all_imgs.reshape(epoch_duration, 1, 1, height, width)
  all_actions = all_actions.reshape(epoch_duration, 1, env.n_actions)

  all_imgs = torch.from_numpy(all_imgs).float()
  all_actions = torch.from_numpy(all_actions).float()
  
  return all_imgs, all_actions


### just go from left to right
def create_epoche_imgs():
  env = CountEnv()

  epoch_duration = 200

  height = env.observation[:,0].size
  width = env.observation[0,:].size

  all_imgs = np.zeros([epoch_duration,height, width])
  all_actions = np.zeros([epoch_duration,env.n_actions])
    
  for t in range(epoch_duration):
    env.update("left")
    all_imgs[t,:,:] = env.observation/255.
    all_actions[t,:] = env.action_onehot
  
  all_imgs = all_imgs.reshape(epoch_duration, 1, 1, height, width)
  all_actions = all_actions.reshape(epoch_duration, 1, env.n_actions)

  all_imgs = torch.from_numpy(all_imgs).float()
  all_actions = torch.from_numpy(all_actions).float()
  
  return all_imgs, all_actions


def envImageAndActionToPytorchFormat(env):

    img = env.observation/255.    
    action = env.action_onehot

    img = img.reshape(1, env.view_size)
    action = action.reshape(1,env.n_actions)
    
    action = torch.from_numpy(action).float()
    img = torch.from_numpy(img).float()

    return img, action

##############################
## ADD TASK LAYER --- modularized
#################################


def add_task_layer(image, img_size, env):
    
    ### Action: one-hot colomns determine - recite, touch, count, give
    #task_matrix = torch.zeros(img_size,img_size)
    #task_ones = torch.ones(img_size)
    #task_matrix[env.task_n, :] = task_ones
    #task_matrix = torch.reshape(task_matrix, [1,img_size,img_size])
    
    #### Object: one-hot colomns determine objects to count - square, events, nothing
    object_matrix = torch.zeros(img_size,img_size)
    object_ones = torch.ones(img_size)
    object_matrix[env.object_n, :] = object_ones
    object_matrix = torch.reshape(object_matrix, [1,img_size,img_size])
    
    #### Quantifier: All, 1, 2, 3, 4... : All uppest left corner, 1 right to it...
    quant_matrix = torch.zeros(img_size,img_size)
    col = env.quant_n%img_size
    row = int(env.quant_n/img_size )
    quant_matrix[row,col] = 1        
    quant_matrix = torch.reshape(quant_matrix, [1,img_size,img_size])    
    
    inp = image.reshape([2,img_size,img_size])  #.reshape([1,200,200])
    #stacked_image = torch.stack([inp[0],inp[1],task_matrix[0], object_matrix[0], quant_matrix[0] ]).reshape(1,5,img_size,img_size)
    
    stacked_image = image.reshape([1,2,img_size,img_size])
    
    return stacked_image
  
  
##############################
## CREATE TASK VECTOR --- modularized
#################################

def create_task_vector(env):
  
    task_vector = torch.zeros(5)
    object_vector = torch.zeros(5)
    quant_vector = torch.zeros(10)
    
    task_vector[env.task_n] = 1
    object_vector[env.object_n] = 1
    quant_vector[env.quant_n] = 1
    
    
    stacked_vector = torch.cat([task_vector, object_vector, quant_vector])
    return stacked_vector



# #######
# # Create the following matrix:
# # All 1  2  3
# #  4  5  6  7
# #  8  9  10 11
# #  12 13 14 15

# quant_matrix = torch.zeros(4,4)

# for i in range(16):
#     col = i%4
#     row = int((i)/4 )
#     quant_matrix[row,col] = i


##############################
## ADD TASK LAYER
#################################


def add_task_layer_old(image, img_size, task_n = 0):
        
    task_matrix = torch.zeros(img_size,img_size)
    task_ones = torch.ones(img_size)
    task_matrix[task_n, :] = task_ones
    task_matrix = torch.reshape(task_matrix, [1,img_size,img_size])
    
    inp = image.reshape([2,img_size,img_size])  #.reshape([1,200,200])
    stacked_image = torch.stack([inp[0],inp[1],task_matrix[0] ]).reshape(1,3,img_size,img_size)
    
    return stacked_image
  

def get_object_features(env):
    object_features = []

    # Add hand features
    object_type = np.array([0,1,0,0])
    feature_vector = np.array([env.hand.pos.x/env.img_size, env.hand.pos.y/env.img_size])
    feature_vector = np.concatenate((feature_vector, object_type), 0) 
    object_features.append( feature_vector )



    # Add square features
    for square in env.squares:
        object_type = np.array([1,0,0,0])
        feature_vector = np.array([square.pos.x/env.img_size, square.pos.y/env.img_size])
        feature_vector = np.concatenate((feature_vector, object_type), 0)
        object_features.append(feature_vector)

    object_features = torch.from_numpy( np.array(object_features)  )
    
    return object_features
  
  
#object_features = get_object_features(env)    
#print(object_features)


def get_dual_relations_from_features(object_features):
  
    b_size = 1
    hidden_size = object_features[0,:].size()[0]   # naming hidden_size comes from convLSTM when number of channels/hidden layers equaled number of features. now should be called: n_features
    n_objects = object_features[:,0].size()[0]
    
    a_flat = object_features.view(b_size, -1, hidden_size)

    a_i = a_flat.unsqueeze(1)
    a_i = a_i.repeat_interleave(repeats=n_objects,dim=1)

    b_j = a_flat.unsqueeze(2)
    b_j = b_j.repeat_interleave(repeats=n_objects,dim=2)

    a_i_exp = a_i.unsqueeze(3)
    b_j_exp = b_j.unsqueeze(3)

    c = torch.cat((a_i_exp, b_j_exp), 3)
    c_final = c.view(b_size, -1, 2*hidden_size)


    return c_final
  
class result_tensor():
  
    def __init__(self):
        self.task = []
        self.n_obj = []
        self.episode = []
        self.accuracy = []
        
        self.losses = []
        self.losses_episodes = []
        self.runs = []
        
        self.n_one_to_ones = []
        self.n_right_number_words = []
        self.variabilities = []
                        
    def add_episode_result(self, task, n_obj, episode, accuracy, run_n,n_one_to_one=None, n_right_number_words=None, variabilities=None):
        self.task.append(task)
        self.n_obj.append(n_obj)
        self.episode.append(episode)
        self.accuracy.append(accuracy)
        self.runs.append(run_n)
        if(n_one_to_one is not None):
          self.n_one_to_ones.append(n_one_to_one)
        if(n_right_number_words is not None):
          self.n_right_number_words.append(n_right_number_words)
        if(variabilities is not None):
          self.variabilities.append(variabilities)        
    def add_loss(self, loss, episode): 
        self.losses.append(loss)
        self.losses_episodes.append(episode)
        
    def create_panda_df(self):
      
        normalized_variabilities = [float(i)/max(self.variabilities) for i in self.variabilities] 
        
        df = pd.DataFrame(
        {
            "task": self.task,
            "n_obj": self.n_obj,
            "episode": self.episode,
            "accuracy": self.accuracy,
            "losses": self.losses,
            "runs": self.runs,
            "n_one_to_ones": self.n_one_to_ones,
            "n_right_number_words": self.n_right_number_words,
            "variabilities": normalized_variabilities
        })
            
        return df



def create_batch(env, batch_size):

    env_img_list = []    # each list item represents one time instance, which contains a whole batch
    env_action_list = []
    env_relation_list = []
    n_list = []
    env_task_list = []
    task_vector_size = env.task_vector_length
    
    


    for n in range(batch_size):
        env.reset()
        env.solve_task()

        

        task_vector = torch.zeros(task_vector_size)
        task_vector[env.task_n] = 1
        
        
        task_vector = create_task_vector(env)
        env_task_list.append(task_vector)
      

        for t in range(0, len(env.epoch)-1 ):
            

            #stacked_img_coord = add_task_layer(env.epoch[t]['img'], env.img_size, env.task_n)
            #stacked_img_coord = add_task_layer(env.epoch[t]['img'], env.img_size, env)
            stacked_img_coord = env.epoch[t]['img']

            
            if(t>=len(env_img_list) ):
                env_img_list.append(stacked_img_coord)
                env_action_list.append(env.epoch[t]['action'])
                #env_relation_list.append(env.epoch[t+1]['rel'])
                
                n_list.append( np.array(n) )
            else:
                env_img_list[t] = torch.cat([env_img_list[t],  stacked_img_coord])
                env_action_list[t] = torch.cat([env_action_list[t],  env.epoch[t]['action']])
                #env_relation_list[t] = env_relation_list[t] #torch.cat([env_relation_list[t],  env.epoch[t+1]['rel']])    !!!!!!! #change
                
                #env_action_list[t] = torch.cat([env_action_list[t],  env.epoch[t+1]['action']])
                #print("ind_list[t]: ", ind_list[t])
                #print("np.where(current_ind_list==n)[0]", np.where(current_ind_list==n))
                n_list[t] = np.append( n_list[t], n )
    # Dummy last entry: won't be needed in training, since after last step state_network not updated anymore
    n_list.append( np.array(n) )
    
    current_ind_list = np.arange(batch_size)
    ind_list = []
        
    #print("n_list ", n_list )

    for t in range(len(n_list) - 1 ):
        #print("ind_list[t]: ", ind_list[t])
        first_n = True
        for n_ in np.ndenumerate(n_list[t+1]):
            n = n_[1]
            appendy = np.where(n_list[t]==n)  
            if type(appendy) is tuple:
                appendy=appendy[0]
            if(first_n):
                ind_list.append( appendy  )
                first_n = False
            else:
                ind_list[t] = np.append(ind_list[t],  appendy  )
            
            #pass
            #print(n)
            #if(np.any(current_ind_list[:, 0] == n))
            #ind_list[t] = current_ind_list[ind_list[t]]
            #current_ind_list = current_ind_list[ind_list[t]]
            
        
        #ind_list.append(np.where(current_ind_list==n)[0]  )
        #ind_list[t] = np.concatenate(ind_list[t], np.where(current_ind_list==n)[0])
    #print("ind_list: ", ind_list)
      
      
      
    for t in range(len(ind_list)):
          ind_list[t] =  torch.from_numpy( ind_list[t] ) 
    
    env_task_list = torch.stack(env_task_list)
    
    
      
    return  env_img_list, env_action_list, ind_list, env_relation_list, env_task_list  
  
