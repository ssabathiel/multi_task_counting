#################################
###### TEST MODEL UUUSED
###########################

print("Load test process .... ")

def test_model(env, model, n_test_runs = 10):
    
      model.eval()

      n_successful_runs = 0
      n_one_to_one_correspondences = 0
      n_right_number_order = 0
      variabilities = 0.0
      task_vector_size = 5
      
      
      for n in range(n_test_runs):
      
          #env = CountEnv(mode="target_space", max_dist = 10, n_squares = 1, display = "None", save_epoch = True )
          #env.task = "touch_all_objects"
          if(n==0):
                env.display = "game"
          else:
                env.display = "None"
          env.reset()
          max_t = env.max_time
          t=0
          #state_network = None
          
          state_network_vis = None
          state_network_lang = None
          input_lang = torch.zeros( (1, env.n_words) )
          task_vector = torch.zeros(task_vector_size)
          task_vector[env.task_n] = 1
          
          task_vector = create_task_vector(env)
          task_vector = task_vector.reshape(1, env.task_vector_length)
          
          a_is = []
      
          while(t<max_t and env.ended == False):
              if(n==0):
                  env.display = "Game"
              t += 1

              image, action = envImageAndActionToPytorchFormat(env)
              #stacked_img_coord = add_task_layer(image, env.img_size, env.task_n)
              #stacked_img_coord = add_task_layer(image, env.img_size, env)
              stacked_img_coord = image
              
              if(CUDA_bool==False): 
                 state_network_vis, output_action, state_network_lang, output_lang = model(stacked_img_coord,state_network_vis, input_lang, state_network_lang, task_vector)
              else:
                  state_network_vis, output_action, state_network_lang, output_lang = model(stacked_img_coord.cuda(),state_network_vis, input_lang.cuda(), state_network_lang, task_vector.cuda())

              #print("###########################")
              #print("whole action: ", torch.cat((output_action, output_lang),1))
              #print("output_lang.detach().numpy(): ", output_lang.detach().numpy() )
              #print("output_action.detach().numpy(): ", output_action.detach().numpy() )
              #print("output_lang.detach().numpy()[0][:-1]: ", output_lang.detach().numpy()[0][:-1] )
              #print("output_lang[0][-1].detach().numpy(): ", output_lang[0][-1].detach().numpy())
              
              
              a = int( np.argmax(output_action.detach().cpu().numpy()[0][:-1]).item() )  # cpu: int( np.argmax(output_action.detach().cpu().numpy()[0][:-1]).item() )
              
              #print(output_lang.detach().numpy()[0][-1])
              Is_a = bool( round( output_action[0][-1].detach().cpu().numpy().item() ) )
              c = bool( round( output_lang[0][-1].detach().cpu().numpy().item() ) )

                  
              word = int(np.argmax(output_lang.detach().cpu().numpy()[0][:-1]).item() )
              
              #if(c):
              #    print(word+1)
              #    print(state_network_lang)
              
              #print("word: ", word)
              #print("a: ", Action[a]  )
              #print("word: ", word)
              #print("Action[word]: ", Action[word])
              #print("c: ", c)
              env.triple_update(int(a),Is_a, int(word+env.n_motor_actions), c )
              #print("c: ", c)
              #c = True
              #input_lang = torch.zeros( env.n_words+1 ) #.view(1, env.n_words+1)
              #if(c==True):
              #  input_lang[word] = 1
              #  input_lang[env.n_words] = 1
                
              #print("input_lang: ", input_lang)
              #input_lang = input_lang.view(1, env.n_words+1)
              input_lang = copy(output_lang)
              input_lang = torch.from_numpy( env.action_onehot[-env.n_words-1:-1] ).float().view(1, -1) # env_action_list[t][-env.n_words:]
              
              # get variability
              action_length = output_action.detach().numpy()[0][:-1].size
              verbal_length = output_lang.detach().numpy()[0][:-1].size
              n_actions = action_length + verbal_length
              
              if(Is_a):
          	     a_is.append(a)
              if(c):
                 a_is.append(word)
              
              
              
              if(env.ended):
                  n_successful_runs += 1
          
          if(env.one_to_one_correspondence==True):
             n_one_to_one_correspondences += 1
          if(env.right_number_order==True):
             n_right_number_order += 1
              
          ### get variability
          f_is = []
          action_sequence_length = len(a_is)
      
          for i in range(n_actions):
             if(action_sequence_length>0):
             	f_is.append(a_is.count(i)/action_sequence_length)
             else:
                f_is.append(0)
      
          sum_of_squared_f_is = 0
          for i in range(len(f_is)):
             sum_of_squared_f_is += f_is[i]*f_is[i]
      
          variability = 1-np.sqrt(sum_of_squared_f_is)
          variabilities += variability
      
      n_spaces = ( 20 - len(env.task) ) * " "            
      print(env.task, " ", "/ n =", env.n_squares, " :", n_spaces , n_successful_runs, " / ", n_test_runs, " test runs successful // 1-1: ", n_one_to_one_correspondences, " / ", n_test_runs, "Number order: ", n_right_number_order, " / ", n_test_runs)
      
      return n_successful_runs, n_one_to_one_correspondences/n_test_runs, n_right_number_order/n_test_runs, variabilities/n_test_runs

    
def test_model_and_write(env, n_test_runs = 10):
    success_number, n_one_to_one_correspondences, n_right_number_order = test_model(env, n_test_runs)
    result_tensory.add_episode_result(task, n_, episode, success_number)
    
    
    
    