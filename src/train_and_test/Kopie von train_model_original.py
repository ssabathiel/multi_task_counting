##########################################
###### SUPERVISED LEARNING - BATCH SIZE - BATCH SIZE
#####################################
#########################
################
##############
###########
##########
#########
########
#######
######
#####
####
###
##
#


print("Load training process....")

def train_model(task_list=["touch_all_objects"], n_squares_ = 1, num_epochs = 2000, episode = 0, run_n = 0, model=None):
  
    model.train()  
    """
    Run some basic tests on the API
    """
    
    #lr = 5e-1
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr = model.lr)
    
    max_epoch = num_epochs  # number of epochs
    test_every_n = 200

    torch.manual_seed(0)
    loss_fn = nn.MSELoss()
    #loss_fn = nn.BCELoss()
    #loss_fn = nn.NLLLoss()
    #loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.softmax_cross_entropy_with_logits()
    
    
    #mode_ = "target_space"
    #n_squares_ = random.randint(1,n_squares_)
    env = CountEnv(task_ = task_list[0], mode= "pick_square", n_squares = n_squares_, display = "None", save_epoch = True )
    env.sample_task = True
    env.rand_n_squares=True
    env.task_list = task_list
    #env.n_squares_max_list[1] = 2    
    
    batch_size = 8*len(task_list)
    
    average_loss = 0
    
    
    #env_img_list, env_action_list, ind_list, _, env_task_list = create_batch(env, 1)
    #print("env_action_list: ", env_action_list)
    
    #################################
    ###### TRAIN
    ###########################
    print('Run for', max_epoch, 'iterations')


    #################################
    ## Initial Test before training
    ##############################
    episode = model.episode
    for task in env.task_list:
          for n_ in range(1, env.n_squares_max+1):
              env.task = task
              env.rand_n_squares = False
              env.sample_task = False
              env.reset()
              env.n_squares_wished = n_
              n_test_runs = 20
              success_number = test_model(env, model, n_test_runs)
              model.result_tensory.add_episode_result(task, n_, episode, success_number/n_test_runs, run_n)
              model.result_tensory.add_loss(average_loss/test_every_n, episode)
              
    env.rand_n_squares = True
    env.sample_task = True
    env.n_squares_wished = -1
    #env.n_squares_wished = env.n_squares
    
       
    
    
    for epoch in range(0, max_epoch):
          episode = model.episode
          model.episode = model.episode + 1
          
          model.lr = max(1e-2,model.lr)
          #if(epoch==3000):
          #    lr = 1e-3
          state_network = None
          
          
          env.reset()
          
          '''
          if(env.task=="touch_all_objects"):
              touch_all_objects(env)
              
          if(env.task=="count_all_objects"):
              count_all_objects(env)     
              
          if(env.task=="move_all_squares_from_source_to_target"):
              move_all_squares_from_source_to_target(env)
          '''   
          
          #env_img_list, env_action_list, ind_list, _, env_task_list = create_batch(env, 1)
          #print("env_action_list: ", env_action_list)
          
          
          ###################################

          state_network_vis = None
          state_network_lang = None

          env_img_list, env_action_list, ind_list, _, env_task_list = create_batch(env, batch_size)
          input_lang = torch.zeros( (batch_size, env.n_words+1) ).view(batch_size, env.n_words+1)
          
          loss = 0

          ##########################
          

          
          for t in range(0, len(env_img_list)):
             
              if(CUDA_bool==False): 
                 state_network_vis, output_action, state_network_lang, output_lang = model(env_img_list[t],state_network_vis, input_lang, state_network_lang, env_task_list) 
              else:
                 state_network_vis, output_action, state_network_lang, output_lang = model(env_img_list[t].cuda(),state_network_vis, input_lang.cuda(), state_network_lang, env_task_list.cuda())
              
              Q_values = torch.cat((output_action, output_lang),1)
              
              if(CUDA_bool==False): 
                 #loss += loss_fn(env_action_list[t], Q_values )
                 #loss += loss_fn(Q_values, torch.max(env_action_list[t], 1)[1] )
                 loss += loss_fn(Q_values, env_action_list[t] )
                  
                 #loss += loss_fn(env_action_list[t], torch.max(Q_values, 1)[1] ) 
                 #loss += own_cross_entropy(env_action_list[t], Q_values)
                 #_, labels = torch.max(Q_values, 1)
                 #loss += loss_fn(env_action_list[t], labels) 
                 #loss += cross_entropy_one_hot(env_action_list[t], Q_values)
                 #     _, labels = target.max(dim=1)
                 #return nn.CrossEntropyLoss()(input, labels)
              else:
                 loss += loss_fn(env_action_list[t].cuda(), Q_values)
              #if(epoch==0):
              #  print("env_action_list[t]: ", env_action_list[t])
              #  print("Q_values ", Q_values)

              input_lang = copy(output_lang)

              
              #print(Q_values.size() )
              #print(env_action_list[t].size())
              #print("state_network[0].size(): ", state_network[0].size())
              #print("state_network[0][ind_list[t]].size(): ", state_network[0][ind_list[t]].size())
              #print(ind_list[t])
              #state_network[0] = state_network[0][ind_list[t]]
              #state_network[1] = state_network[1][ind_list[t]]
              
              state_network_vis[0] = state_network_vis[0][ind_list[t]] #.detach()   #if you want to retain graph/LSTM-cell
              state_network_vis[1] = state_network_vis[1][ind_list[t]] #.detach()
              
              state_network_lang[0] = state_network_lang[0][ind_list[t]] #.detach()
              state_network_lang[1] = state_network_lang[1][ind_list[t]] #.detach()
              
              input_lang = input_lang[ind_list[t]]
              env_task_list = env_task_list[ind_list[t]]
          
              #if((t+1)%20==0 or t==(max_epoch-1) ):
              #print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data) )
              
              if( (t+1)%1000==0):
                 model.zero_grad()
                  
                 loss.backward(retain_graph=True)
                 optimizer.step()
              
          average_loss += loss.data

          # zero grad parameters
          model.zero_grad()
          # compute new grad parameters through time!
          loss.backward()
          optimizer.step()
          # learning_rate step against the gradient
          #for p in model.parameters():
          #p.data.sub_(p.grad.data * lr)
              
 
            
          if((epoch+1)%test_every_n==0):
              print("Episode: ", epoch+1)
              print("Learning rate: ", get_lr(optimizer))
              print("Average Loss in past ", test_every_n, " runs: ", average_loss/test_every_n)
          
              '''
              env.sample_task = False
              with ThreadPoolExecutor(max_workers=2) as executor:
              for task in env.task_list:
                  env.task = task
                  env.reset()
                  success_number = executor.submit(test_model, env, n_test_runs = 50)
                  success_number_list_count_events.append(success_number/50.)
              '''
              
              for task in env.task_list:
                  print(" ")
                  #if(model.model_path != ""):
                  #  all_text_path = model.model_path + "GIFs_and_ACTIONS/actions.txt"
                  #  whole_text = "\n" + "Episode " + str(model.episode) + "\n"
                  #  fily = open(all_text_path,"a+")
                  #  fily.write(whole_text)
                  #  fily.close()
                  
                  for n_ in range(1, env.n_squares_max+1):
                      env.task = task
                      env.rand_n_squares = False
                      env.sample_task = False
                      env.reset()
                      env.n_squares_wished = n_
                      n_test_runs = 20
                      success_number = test_model(env, model, n_test_runs)
                      model.result_tensory.add_episode_result(task, n_, episode, success_number/n_test_runs, run_n)
                      model.result_tensory.add_loss((average_loss/test_every_n).numpy(), episode)
                      
                      # Save GIF and ACTION-sequence:
                      
                      if(model.model_path != ""):
                        env.reset()
                        env.n_squares_wished = n_
                        n_test_runs = 20
                        directory_path = model.model_path + "GIFs_and_ACTIONS/"+ str(model.episode) + "/"
                        #if(not os.path.exists(directory_path) ):
                        #os.mkdir(directory_path)
                        file_name = "__" + env.task + "__" + str(env.n_squares_wished)
                        pathy = directory_path + file_name
                        
                        
                        demonstrate_model(env,model, PATH = pathy)
                      

              env.rand_n_squares = True
              env.sample_task = True
              env.n_squares_wished = -1
              '''
              with ThreadPoolExecutor(max_workers=25) as executor:
                  for task in env.task_list:
                      print(" ")
                      for n_ in range(1, env.n_squares_max+1):
                          env.task = task
                          env.rand_n_squares = False
                          env.sample_task = False
                          env.reset()
                          env.n_squares_wished = n_
                          success_number = executor.submit(test_model_and_write, env, n_test_runs = 50)
                                        
              '''

              
              print("---------------------------------------")
              if(average_loss/test_every_n < 0.2):
                 model.lr = 0.05 #0.95
              for g in optimizer.param_groups:
                  g['lr'] = model.lr
              
              
              #average_loss_list.append(1.0*average_loss/test_every_n)
              #episode_list.append(episode+epoch)
              
              average_loss = 0
              
          
         
task_setup = {
    'task': "touch_all_objects",
    'max_dist': 10,
    'n_squares': 1,
    'num_epochs': 600
}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

      
def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)
  
def own_cross_entropy(inputy, target):
  logsoftmax = nn.LogSoftmax()
  return torch.mean(torch.sum(-target * logsoftmax(inputy), dim=1))
  
  
#train_model(task=["count_all_objects"], n_squares_ = 4, num_epochs = 2500, episode = episode) 
#train_model(task="move_all_squares_from_source_to_target", max_dist_ = 15, n_squares_ = 3, num_epochs = 200)
#train_model(task="pick_next_object", max_dist_ = 10, n_squares_ = 2, num_epochs = 2000) 
   

             