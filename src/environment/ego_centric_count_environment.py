
print("Load Count-Environment..")

class Square():
  def __init__(self, pos, id_, n_neighbours):
    self.pos = pos
    self.id = id_
    self.n_neighbours = n_neighbours
    self.update_obs()
    
    self.picked_already = False 
    self.touched_already = False 
    
    
  def update_obs(self):
    self.data = np.ones((1, 1), dtype=np.uint8)*255
    self.obj_belong_fct = np.zeros((2*self.n_neighbours, 2*self.n_neighbours), dtype=np.uint8)
    

    
  def move(self, direction):
    move_dist=1
    
    if(direction=="right"):
      self.pos.x += move_dist
    elif(direction=="left"):
      self.pos.x -= move_dist
    elif(direction=="up"):
      self.pos.y += move_dist
    elif(direction=="down"):
      self.pos.y -= move_dist
      
    self.update_obs()

class Hand():
    def __init__(self, data,data_mask, pos):
      self.data = data
      self.pos = pos
      self.data_mask = data_mask
      

class Pos():
  def __init__(self, x_, y_):
    self.x = x_
    self.y = y_
  
  
class CountEnv():
    def __init__(self,task_ = "touch_all_objects", mode="pick_square", max_dist = 20, n_squares = 1, display = "None", save_epoch = False, img_size = 4, obj_source="squares"):
        
        self.img_size = img_size
        self.view_size = 7
        self.mode = mode
        self.n_squares_max = n_squares
        self.max_dist = max_dist
        self.task = task_   # move_all_squares_from_source_to_target / touch_all_objects
        
        self.count_action = False
        self.motor_action = False
        self.since_count_action = 0
        self.show_number_length = 1
        self.show_number = False
        self.last_count_number = 0
        self.rand_n_squares=True
        
        self.IsTripleAction = False
        self.action_motor = ""
        self.action_IsSayWord = True
        self.action_word = ""
        
        self.counted_word_list = [] 
        self.aimed_count_list = []
        self.counted_square_list = []
        self.aimed_given_square_id_list = []
        self.given_square_id_list = []
        
        self.sample_task = False
        self.task_list = None
        self.observation_hand = []
        self.observation_square = []
        
        self.n_squares_max_list = [self.n_squares_max]*10
        self.n_squares_wished = -1
        
        
        
        
        self.reset()
        
        self.n_motor_actions = 7
        self.n_words = 10
        
        self.n_actions = self.n_motor_actions + self.n_words + 2
        self.action = 1
        self.action_onehot = np.array([int(i == 0) for i in range(self.n_actions)])
        self.total_reward = 0
        self.display = display
        self.save_epoch = save_epoch
        self.relations = []
        
        self.move_dist = 1

        
        self.task_vector_length = 20
        self.obj_source = obj_source
        
        self.pick_from_00 = False
        self.pick_from_00_then_move = False
        self.give_n_current_squares = 1
        self.readable_action_onehot = []
        
        self.print_action_onehot_aftersteps = False
        
        

            
        
        

        
        
    def reset(self):
      
        self.time = 0
        self.ended = False
        self.total_reward = 0
        self.picked = False
        self.time_penalty = 0.02
        self.epoch = []
        self.IsTripleAction = False
        self.hand_is_visible = True
        self.square_is_visible = True
        self.second_action = True
        self.since_last_event_count = 1
        self.event_had_occured_already = False
        self.last_action_is_count = True
        self.stopped_early = False
        
        self.n_wait_steps = random.randint(1,5)
        self.last_event_count = 0
        
         
        
        if(self.sample_task==True):
            self.task = random.sample(self.task_list,1)[0]  
        
        if(self.task_list is None):
          self.task_n = 0
        else:
          self.task_n = self.task_list.index(self.task)
            
            
        if(self.rand_n_squares==True):
            self.n_squares = random.randint(1,self.n_squares_max)
            self.n_squares = random.randint(1,self.n_squares_max_list[self.task_n])
        else:
            self.n_squares = self.n_squares_max
        
        if(self.rand_n_squares==True and self.sample_task==True):
            self.n_squares = random.randint(1,self.n_squares_max_list[self.task_n])

        if(self.n_squares_wished >= 0):
            self.n_squares = self.n_squares_wished
        
        self.task_n = 0
        
        self.counted_word_list = [] 
        self.aimed_count_list = []
        self.counted_square_list = []
        self.given_square_id_list = []
        self.aimed_given_square_id_list = []
        
        for i in range(self.n_squares):
            self.aimed_count_list.append(str(i+1) )
            self.aimed_given_square_id_list.append(str(i+1) )
            


        self.max_time = 100
        ## Task encoding: task, object and quantifier to integer
        if(self.task == 'touch_all_objects'):
            self.task_n = 0
            self.object_n = 0
            self.quant_n = 0
            self.max_time = self.n_squares*10
        #elif(self.task == 'move_all_squares_from_source_to_target'):
        #    self.task_n = 1
        #    self.object_n = 0
        #    self.quant_n = 0
        elif(self.task == 'count_all_objects'):
            self.task_n = 1
            self.object_n = 0
            self.quant_n = 0
            self.max_time = self.n_squares*10
        elif(self.task == 'count_all_events'):
            self.task_n = 1
            self.object_n = 1
            self.quant_n = 0
            self.hand_is_visible = False
            self.max_time = self.n_squares*8
            
        elif(self.task == 'give_n'):
            self.task_n = 2
            self.object_n = 0
            self.quant_n = self.n_squares  
            self.max_time = self.n_squares*9 + 6  
                        
        elif(self.task == 'recite_n'):
            self.task_n = 3
            self.object_n = 0
            self.quant_n = self.n_squares
            self.hand_is_visible = False
            self.square_is_visible = False
            self.max_time = self.n_squares*2
            
        elif(self.task == 'do_nothing'):
            self.task_n = 4
            self.object_n = 1
            self.quant_n = 0
            self.hand_is_visible = False
            self.square_is_visible = False
            self.max_time = self.n_squares*2
            self.did_nothing = True

        elif(self.task == 'recite_n_inverse'):
            self.task_n = 5
            self.object_n = 0
            self.quant_n = self.n_squares
            self.hand_is_visible = False
            self.square_is_visible = False
            self.max_time = self.n_squares*2
            for i in range(self.n_squares):
               self.aimed_count_list.append(str(self.n_squares-i) )
           
        
            

            
        background = np.zeros([self.img_size,self.img_size])
        
                
        self.background = background
        
        
        pnt_size = 1
        
        pointer, pointer_mask = np.array([[255]]), np.array([[255]])
        pointer_grab, _ = np.array([[255]]), np.array([[255]])

        hand_pos_x = random.randint(0,self.img_size-1) 
        hand_pos_y = random.randint(0,self.img_size-1) 
        pos = Pos(hand_pos_x, hand_pos_y)
        hand = Hand(pointer,pointer_mask, pos)
        hand_grab = Hand(pointer_grab,pointer_mask, pos)
        
        squares = [] #Create_N_Sqaures(self.n_squares, mode = self.mode, max_dist = self.max_dist, img_size = self.img_size)
        pos_list = []
        for n in range(self.n_squares):
            
            pos_not_ok = True
            rand_pixel_1 = random.randint(0,self.img_size-1) 
            rand_pixel_2 = random.randint(0,self.img_size-1) 
            
            while(pos_not_ok):
                rand_pixel_1 = random.randint(0,self.img_size-1) 
                rand_pixel_2 = random.randint(0,self.img_size-1) 
                pos_array = np.array([rand_pixel_1, rand_pixel_2])
                
                if(any((pos_array == x).all() for x in pos_list)):
                    pos_not_ok = True
                else:
                    pos_not_ok = False
                    pos_list.append(pos_array)
                                      
            pos = Pos(rand_pixel_1, rand_pixel_2)
            square_now = Square(pos, n+1, 0 )  #data, pos, id_, n_neighbours
            squares.append(square_now)
        
        if(self.task == "give_n"):
            self.squares = []
            pos = Pos(0, 0)
            square_now = Square(pos, 1, 0 )  #data, pos, id_, n_neighbours
            self.squares.append(square_now)
            self.obj_source = "infinite_squares"
            #self.squares = squares
        else:
            self.squares = squares
        
        if(self.task=="count_all_events"):
            self.squares = []
            pos = Pos(3, 3)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)
            self.event_there = False
            self.square_is_visible = False
        
        
        
        self.hand = hand        
        self.hand_nongrab = copy(hand)
        self.hand_grab = copy(hand_grab)
        
        self.visObs, self.observation, self.obj_belong_fct = self.constructObs()
        #print("self.obj_belong_fct: ", self.obj_belong_fct)
        self.IsGrab = False
        self.IsTouch = False
        
        self.grabed_square = 0
        
        self.reward = 0
        self.picked_once_already = False
        self.been_right_already = False
        self.is_done = False
        #display(Image.fromarray(hand.data) )
        #display(Image.fromarray(hand_grab.data) )
        self.pick_from_00 = False
        self.pick_from_00_then_move = False
        self.give_n_current_squares = 1
        
        
    def constructObs(self):
            
        # Array
        #self.observation = copy(self.background)
        self.observation = np.zeros([self.img_size,self.img_size])
        self.observation_hand = copy(self.background)
        self.observation_square = copy(self.background)
        self.obj_belong_fct = np.zeros([self.img_size,self.img_size]).astype(int) #copy(self.background).astype(int)

        if(self.square_is_visible == True):
            for square in self.squares:     
              #print("in there")
              #Array
              foreground_square = square.data  
              self.observation[square.pos.x-square.n_neighbours:square.pos.x-square.n_neighbours + square.data.shape[0], square.pos.y-square.n_neighbours:square.pos.y-square.n_neighbours + square.data.shape[1]] = foreground_square
              self.observation_square[square.pos.x-square.n_neighbours:square.pos.x-square.n_neighbours + square.data.shape[0], square.pos.y-square.n_neighbours:square.pos.y-square.n_neighbours + square.data.shape[1]] = foreground_square

              foreground_belong = (square.data/255)*square.id
              #self.obj_belong_fct[square.pos.x-square.n_neighbours:square.pos.x-square.n_neighbours + square.data.shape[0], square.pos.y-square.n_neighbours:square.pos.y-square.n_neighbours + square.data.shape[1]] = foreground_belong.astype(int)
              if(square.pos.x<self.img_size and square.pos.x>=0):
                 if(square.pos.y<self.img_size and square.pos.y>=0):  
                    self.obj_belong_fct[square.pos.x, square.pos.y] = square.id #foreground_belong.astype(int)
          
        #self.observation[self.hand.pos.x, self.hand.pos.y] = 255
        
        observation_copy = copy(self.observation)  
        
        if(self.hand_is_visible == True):
              if(self.hand.pos.x<self.img_size and self.hand.pos.x>=0):
                 if(self.hand.pos.y<self.img_size and self.hand.pos.y>=0):
                    observation_copy[self.hand.pos.x, self.hand.pos.y] = 127        
                    self.observation_hand[self.hand.pos.x, self.hand.pos.y] = 255
        #self.observation[self.hand.pos.x:self.hand.pos.x + self.hand.data.shape[0], self.hand.pos.y:self.hand.pos.y + self.hand.data.shape[1]] = self.hand.data
        
        
        ###############################
        ## EGO-CENTRIC VIEW
        observation_copy_mean_while = np.zeros((7,7))
        observation_copy_square_mean_while = np.zeros((7,7))
        observation_copy_hand_mean_while = np.zeros((7,7))
        
        ## Updating whole image presented
        #observation_copy_mean_while[:observation_copy.shape[0],:observation_copy.shape[1]] = observation_copy
        #observation_copy_mean_while[self.hand.pos.x:self.hand.pos.x + observation_copy.shape[0],self.hand.pos.y:self.hand.pos.y + observation_copy.shape[1]] = observation_copy
        observation_copy_mean_while[3-self.hand.pos.x:3-self.hand.pos.x + observation_copy.shape[0],3-self.hand.pos.y:3-self.hand.pos.y + observation_copy.shape[1]] = observation_copy
        
        ## Updating square observation
        observation_copy_square_mean_while[3-self.hand.pos.x:3-self.hand.pos.x + self.observation_square.shape[0],3-self.hand.pos.y:3-self.hand.pos.y + self.observation_square.shape[1]] = self.observation_square
        
        ## Updating hand observation
        observation_copy_hand_mean_while[3-self.hand.pos.x:3-self.hand.pos.x + self.observation_hand.shape[0],3-self.hand.pos.y:3-self.hand.pos.y + self.observation_hand.shape[1]] = self.observation_hand
        
        observation_copy = observation_copy_mean_while
        self.observation_square = observation_copy_square_mean_while
        self.observation_hand = observation_copy_hand_mean_while
        

                
        #self.observationImg.paste(hand_img, (self.hand.pos.x, self.hand.pos.y), hand_mask)
        self.observationImg = Image.fromarray(observation_copy)
        self.observationImg = self.observationImg.resize( (400,400))
        
                #### Count window
        if(self.show_number == True):
            count_window_size = 8
            intensity_factor = 1 # np.sin(self.since_count_action/self.show_number_length*np.pi)
            count_window = create_count_window(count_window_size, n=self.last_count_number )*intensity_factor
            count_window = Image.fromarray(count_window).resize((50,50)).convert('RGB')
            bord_dist = int(self.img_size/20)       
            #observation_copy[bord_dist:count_window_size+bord_dist, self.img_size - count_window_size - bord_dist:self.img_size-bord_dist] = count_window
            #self.observation[bord_dist:count_window_size+bord_dist, self.img_size - count_window_size - bord_dist:self.img_size-bord_dist] = count_window 
            bg_w, bg_h = self.observationImg.size
            img_w, img_h = count_window.size
            offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
            if(self.task == "count_all_events"):
              offset = ((bg_w - img_w) // 12, (bg_h - img_h) // 12)
            self.observationImg.paste(count_window, offset)
        
        return self.observationImg, self.observation, self.obj_belong_fct
      
    def solve_task(self):
                 
          if(self.task == "touch_all_objects"):
              touch_all_objects(self)
              
          if(self.task == "count_all_objects"):
              count_all_objects(self)              
          if(self.task == "move_all_squares_from_source_to_target"):
              move_all_squares_from_source_to_target(self)
          if(self.task == "give_n"):
              give_n(self) 
          if(self.task == "count_all_events"):
              count_all_events(self) 
          if(self.task == "recite_n"):
              recite_n(self)  
          if(self.task == "do_nothing"):
              do_nothing(self)    
          if(self.task == "recite_n_inverse"):
              recite_n_inverse(self)    
              
    def print_actions(self):
        a = self.action_onehot.astype(int).astype(str).tolist()
        b = self.a_strings[0]
        
        print("--------------------------------")
        print(a)
        print(b)
        print("--------------------------------")
        
    def readable_actionstring(self):
        a = self.action_onehot.astype(int).astype(str).tolist()
        b = self.a_strings[0]
        
        readable_actionstring = str(a) + "\n" + str(b)

        return readable_actionstring
    
    def triple_action_one_hot(self, action_motor,IsDoMotorAction,action_word, action_IsSayWord):
        action_onehot = np.zeros(self.n_actions)
        
        
        if(IsDoMotorAction==True):  
            action_onehot[Action_inv[action_motor]] = 1
            #print("Action_inv[action_motor]: ", Action_inv[action_motor])
            #action_onehot[-2] = 1
            action_onehot[self.n_motor_actions] = 1
        
        
        if(action_IsSayWord):
            action_onehot[Action_inv[action_word]+1] = 1
            #action_onehot[0] = 1  #"+1" because Is_Action sneaks in into onehot
            action_onehot[-1] = 1

            
        self.a_strings = [["D", "U", "R", "L", "P", "Dr", "T","A", "E", "1", "2", "3","4","5","6","7","8","9","S"]]
        self.readable_action_onehot = np.vstack((action_onehot.astype(str),np.asarray(self.a_strings,str)))
        

            
        return action_onehot
              
    def triple_update(self, action_motor, IsDoMotorAction, word, IsSayWord):


              
        # If input was string convert to int
        if(type(action_motor)==int):
            action_motor = Action[action_motor]
        else:
            action_motor = action_motor
            
        if(type(word)==int):
            word = Action[word]
        else:
            word = word
      
        # Globalize to instance variables
        self.action_motor = action_motor
        self.action_IsSayWord = IsSayWord
        self.IsSayWord = IsSayWord
        self.IsDoMotorAction = IsDoMotorAction
        self.action_word = word
        
        self.IsTripleAction = True
        
        self.second_action = False
        
        if(IsDoMotorAction):
          if(IsSayWord == False):
              self.second_action = True
          self.update(action_motor)
        
        self.observationImg_between = self.observationImg
        
        self.second_action = True
        if(IsSayWord):
            self.update(word)
        '''    
        # COUNT ALL EVENTS      
        if(self.task=="count_all_events" and self.second_action == True):        
          self.last_event_count+=1
        
          if(self.last_event_count>self.n_wait_steps):
            self.event_there = True  
            self.event_had_occured_already = True
            self.n_wait_steps = random.randint(1,5)
            self.last_event_count = 0
            self.square_is_visible = True

            self.squares = []
            pos = Pos(1, 1)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)

            pos = Pos(2, 1)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)

            pos = Pos(1, 2)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)

            pos = Pos(2, 2)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)
 
          else:
            self.square_is_visible = False
            self.squares = []
            pos = Pos(3, 3)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square) 
            self.event_there = False
            
          self.since_last_event_count = self.last_event_count
          self.one_step_after_event = False
          if(self.last_event_count == 1 and self.event_had_occured_already == True):
            self.one_step_after_event = True          
        '''
        
        self.action_onehot = self.triple_action_one_hot(self.action_motor, self.IsDoMotorAction, self.action_word, self.IsSayWord)
        #action_motor,IsDoMotorAction,action_word, action_IsSayWord)
          
        
        
        

              
        self.IsTripleAction = False
        self.IsTouch = False
        
        if(self.print_action_onehot_aftersteps):
          self.print_actions()

        if(self.save_epoch):
            img, action = envImageAndActionToPytorchFormat(self)
            object_features = get_object_features(self)
            dual_relations = get_dual_relations_from_features(object_features).float()
            self.relations = dual_relations
            action_string = self.readable_actionstring()
            curr_exp = {'img': img, 'action': action, 'rel': dual_relations, 'dem_img': self.observationImg, 'action_string': action_string}
            self.epoch.append(curr_exp)
            
        self.visObs = self.constructObs()
        
        
        #Turn this on if you want to save the last frame in save_epoch as well for demonstration or anything

        if(self.save_epoch and self.ended):
            img, action = envImageAndActionToPytorchFormat(self)
            object_features = get_object_features(self)
            dual_relations = get_dual_relations_from_features(object_features).float()
            self.relations = dual_relations
            action_string = self.readable_actionstring()
            curr_exp = {'img': img, 'action': action, 'rel': dual_relations, 'dem_img': self.observationImg, 'action_string': action_string}
            self.epoch.append(curr_exp)
        
        
        
        
    def update(self, action):
      
      
      
      #if(self.time > 20):
        #self.ended = True
        
      
      
      self.reward = 0
      
      move_dist=self.move_dist
      
      if(type(action)==int):
        self.action = Action[action]
      else:
        self.action = action
      
      
      
      if(Action_inv[self.action]>7 and Action_inv[self.action]<19):
          self.action_onehot = np.array([int(i == Action_inv[self.action]) for i in range(self.n_actions)])
          self.count_action = True
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = int(self.action)
          self.motor_action = False

      else:
          self.count_action = False
          self.motor_action = True
      
      if(Action_inv[self.action]==4):
          #self.count_action = True
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = 10     
      if(Action_inv[self.action]==5):
          #self.count_action = True
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = 11  
          
      if(Action_inv[self.action]==6):
          #self.count_action = True
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = 12 
          
      if(Action_inv[self.action]==7):
          #self.count_action = True
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = 13 
          
      if(self.since_count_action < self.show_number_length):
          self.since_count_action += 1
      else: 
          self.show_number = False
      #print("incoming action int: ", action)
      #print("working internally with action: ", self.action)
      

      
      
      #################
      ## ACTIONS
      #################
      
      if(self.action=="down"):
        self.action_onehot = np.array([int(i == 0) for i in range(self.n_actions)])
        
        actual_move_distance = 0
        if(self.hand.pos.x<self.img_size-1):
          actual_move_distance = move_dist
          
        # Move hand and object (if grabbed) by actual_move_distance
        self.hand.pos.x += actual_move_distance                
        if(self.IsGrab == True):
          self.squares[self.grabed_square-1].pos.x += actual_move_distance
        
        
      elif(self.action=="up"):
        self.action_onehot = np.array([int(i == 1) for i in range(self.n_actions)])
        
        actual_move_distance = 0
        if(self.hand.pos.x>0):
          actual_move_distance = move_dist
          
        self.hand.pos.x -= actual_move_distance
        if(self.IsGrab == True):
          self.squares[self.grabed_square-1].pos.x -= actual_move_distance
          
      elif(self.action=="right"):
        self.action_onehot = np.array([int(i == 2) for i in range(self.n_actions)])
              
        actual_move_distance = 0
        if(self.hand.pos.y<self.img_size-1):
          actual_move_distance = move_dist
          
        self.hand.pos.y += actual_move_distance
        if(self.IsGrab == True):
          self.squares[self.grabed_square-1].pos.y += actual_move_distance
          
          
      elif(self.action=="left"):
        self.action_onehot = np.array([int(i == 3) for i in range(self.n_actions)])
                
        actual_move_distance = 0
        if(self.hand.pos.y>0):
          actual_move_distance = move_dist
          
        self.hand.pos.y -= actual_move_distance
        if(self.IsGrab == True):
          self.squares[self.grabed_square-1].pos.y -= actual_move_distance
          
          
      elif(self.action=="pick"):
        self.action_onehot = np.array([int(i == 4) for i in range(self.n_actions)])
        #print("self.obj_belong_fct[self.hand.pos.x, self.hand.pos.y] ", self.obj_belong_fct[self.squares[0].pos.x+1, self.squares[0].pos.y+1])
        if(self.obj_belong_fct[self.hand.pos.x, self.hand.pos.y] != 0):
          self.grabed_square = self.obj_belong_fct[self.hand.pos.x, self.hand.pos.y]
          #print("grabed square ", self.grabed_square)
          self.IsGrab = True
          self.hand.data = copy(self.hand_grab.data)
          self.squares[self.grabed_square-1].picked_already = True
          if(self.hand.pos.x == 0 and self.hand.pos.y==0):
            self.pick_from_00 = True
          

          if(self.picked_once_already == False):
            
            self.picked = True
            self.time_penalty = 0.0
          self.picked_once_already = True
        
      elif(self.action=="release"):
        self.action_onehot = np.array([int(i == 5) for i in range(self.n_actions)])
        if(self.IsGrab):
          if(self.hand.pos.y > int(self.img_size/2) ):
            #self.reward += 1.0
            self.is_done = True
        if(self.IsGrab==True and self.hand.pos.y == self.img_size-1):
            self.given_square_id_list.append(str(self.grabed_square) )
            
        self.IsGrab = False  
        self.hand.data = copy(self.hand_nongrab.data)
       
      elif(self.action=="touch"):     
          self.action_onehot = np.array([int(i == 6) for i in range(self.n_actions)])
        
          if(self.obj_belong_fct[self.hand.pos.x + int(self.hand.data.shape[0]/2), self.hand.pos.y + int(self.hand.data.shape[1]/2)] != 0):
              self.touched_square = self.obj_belong_fct[self.hand.pos.x + int(self.hand.data.shape[0]/2), self.hand.pos.y + int(self.hand.data.shape[1]/2)]
              self.IsTouch = True
              self.hand.data = copy(self.hand_grab.data)
              self.squares[self.touched_square-1].touched_already = True 
              
      elif(self.action=="stop"):     
        self.action_onehot = np.array([int(i == 7) for i in range(self.n_actions)])
        
      if(self.task == "give_n"):   
          if(Action_inv[self.action]>self.n_motor_actions and Action_inv[self.action]<17):
            self.counted_word_list.append(self.action)
          if(self.action == "stop"):
              if(self.given_square_id_list != self.aimed_given_square_id_list or self.counted_word_list!=self.aimed_count_list): 
                 self.stopped_early = True
          if(self.given_square_id_list == self.aimed_given_square_id_list and self.counted_word_list==self.aimed_count_list and self.action=="stop" and self.stopped_early == False): 
              self.ended = True
        
      ######################
      ## Create action_onehot
      ######################+
      
      if(self.IsTripleAction == False):
        self.IsDoMotorAction = False
        self.IsSayWord = False
      
      # Find out if action is motor action 
      if(Action_inv[self.action]<self.n_motor_actions):
          #self.action_onehot[self.n_motor_actions] = 1
          self.IsDoMotorAction = True
          
       
      #Find out if action is saying word
      if(Action_inv[self.action]>self.n_motor_actions-1):
          #self.action_onehot[self.n_motor_actions] = 1
          
          self.IsSayWord = True
          
          
      self.action_onehot = self.triple_action_one_hot(self.action, self.IsDoMotorAction, self.action, self.IsSayWord)
      if(self.print_action_onehot_aftersteps and self.IsTripleAction==False):
          self.print_actions()
      
      
      

      ### For RL --> rewards
      self.reward -= self.time_penalty
      self.total_reward += self.reward
      
      

    
    
      ##################
      ## UPDATE ENVIRONMENT
      #####################
      
      # GIVE-N
      if( int(self.action == Action_inv[self.action]) < 4 and self.pick_from_00 == True and self.obj_source == "infinite_squares"):
          pos = Pos(0, 0)
          self.give_n_current_squares += 1
          new_square = Square(pos, self.give_n_current_squares, 0 )  #(data,) pos, id_, n_neighbours
          self.squares.append(new_square)
          self.pick_from_00 = False
      
      if( self.pick_from_00==True and self.action == "release"):
          self.pick_from_00 = False
       
      
      # COUNT ALL EVENTS      
      if(self.task=="count_all_events" and self.second_action == True):        
        self.last_event_count+=1
        

        

        if(self.last_event_count>self.n_wait_steps):
            self.event_there = True  
            self.event_had_occured_already = True
            self.n_wait_steps = random.randint(1,5)
            self.last_event_count = 0
            self.square_is_visible = True

            self.squares = []
            pos = Pos(1, 1)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)

            pos = Pos(2, 1)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)

            pos = Pos(1, 2)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)

            pos = Pos(2, 2)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square)
 
        else:
            self.square_is_visible = False
            self.squares = []
            pos = Pos(3, 3)
            new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
            self.squares.append(new_square) 
            self.event_there = False
            
        self.since_last_event_count = self.last_event_count
        self.one_step_after_event = False
        if(self.last_event_count == 1 and self.event_had_occured_already == True):
            self.one_step_after_event = True
        
        
        
      ###########################
      ### SET END CONDITIONS
      ###########################
      if(self.task == "count_all_objects"):
            #print("self.IsTouch: ", self.IsTouch)        
            if(self.IsTouch==True and Action_inv[self.action]>self.n_motor_actions and Action_inv[self.action]<17):
                self.counted_square_list.append(self.touched_square)
                self.counted_word_list.append(self.action)
                #print("counted word list updated")
        
            counted_every_square_exactly_once = True
            for i in range(len(self.counted_square_list)):
                if(self.counted_square_list.count(i+1) !=1  ):
                    counted_every_square_exactly_once = False
                    
            right_count_sequence = False
            if(self.counted_word_list==self.aimed_count_list):
                right_count_sequence = True
              
            if(counted_every_square_exactly_once and right_count_sequence):
              self.ended = True
    
      if(self.task == "touch_all_objects"):
            all_squares_touched = True
            for square in self.squares:
                if(square.touched_already == False):
                    all_squares_touched = False
            if(all_squares_touched):
              self.ended = True
              
      if(self.task == "move_all_squares_from_source_to_target"):
            all_squares_in_target = True
            for square in self.squares:
                if(square.pos.y < int(self.img_size*0.75)):
                    all_squares_in_target = False
            if(all_squares_in_target):
              self.ended = True      

      if(self.task == "count_all_events"):
                    
            if(Action_inv[self.action]>self.n_motor_actions and Action_inv[self.action]<17):
                if(self.one_step_after_event == True):
                    self.counted_word_list.append(self.action)
                    
            right_count_sequence = False
            if(self.counted_word_list==self.aimed_count_list):
                right_count_sequence = True
              
            if(right_count_sequence):
              self.ended = True        
              
      if(self.task == "recite_n" or self.task == "recite_n_inverse"):
                    
            if(Action_inv[self.action]>self.n_motor_actions and Action_inv[self.action]<17):
                if(self.last_action_is_count == True):
                    self.counted_word_list.append(self.action)
                else:
                    self.last_action_is_count = False
                    
            right_count_sequence = False
            if(self.counted_word_list==self.aimed_count_list):
                right_count_sequence = True
              
            if(right_count_sequence and self.action=="stop"):
              self.ended = True  
              
      if(self.task == "do_nothing"):
        if(self.action != "stop"):
           self.did_nothing = False
        if(self.time>self.n_squares-2 and self.did_nothing):
           self.ended = True
      
      if(self.IsTripleAction == False):
          if(self.save_epoch):
              
              img, action = envImageAndActionToPytorchFormat(self)
              object_features = get_object_features(self)
              dual_relations = get_dual_relations_from_features(object_features).float()
              self.relations = dual_relations
              action_string = self.readable_actionstring()
              curr_exp = {'img': img, 'action': action, 'rel': dual_relations, 'dem_img': self.observationImg, 'action_string': action_string}
              self.epoch.append(curr_exp)
              
          self.visObs = self.constructObs()
          
      
        #Turn this on if you want to save the last frame in save_epoch as well for demonstration or anything
      if(self.IsTripleAction == False):
         if(self.save_epoch and self.ended):
             img, action = envImageAndActionToPytorchFormat(self)
             object_features = get_object_features(self)
             dual_relations = get_dual_relations_from_features(object_features).float()
             self.relations = dual_relations
             action_string = self.readable_actionstring()
             curr_exp = {'img': img, 'action': action, 'rel': dual_relations, 'dem_img': self.observationImg, 'action_string': action_string}
             self.epoch.append(curr_exp)
              
      if(self.IsTripleAction == False):
        self.time += 1
      else:
        if(self.second_action == True):
          self.time += 1
    
    

    
Action = {
    0: "down",
    1: "up",
    2: "right",
    3: "left",
    4: "pick",
    5: "release",
    6: "touch",
    7: "stop",
    8: "1",
    9: "2",
    10: "3",
    11: "4",
    12: "5",
    13: "6",
    14: "7",
    15: "8",
    16: "9"
}        

Action_inv = {
    "down": 0,
    "up": 1,
    "right": 2,
    "left": 3,
    "pick": 4,
    "release": 5,
    "touch": 6,
    "stop": 7,
    "1": 8,
    "2": 9,
    "3": 10,
    "4": 11,
    "5": 12,
    "6": 13,
    "7": 14,
    "8": 15,
    "9": 16
}  

readable_task = {
    "touch_all_objects": "Touch all objects",
    "count_all_objects": "Count all objects",
    "count_all_events": "Count all events",
    "give_n": "Give N",
    "recite_n": "Recite N",   
    "do_nothing": "Do nothing",
    "recite_n_inverse": "Recite N inverse"
}


def create_pointer(width, height, pick=False):
  #width += 1
  #height += 1
    
  data = np.zeros((width,height), dtype=np.uint8)
  data_mask = np.zeros((width,height), dtype=np.uint8)
  
  for i in range(width):
    for j in range(height):
      h = width - i -1
      w = width - j -1
      
      strips = h
      if(pick):
        strips = w
      
      if(abs(j)<= width/2.0 ):
        if(h <= j or h == j or h == j+2):
          data_mask[i,j] = 255
        if(h <= j and strips%2==0 or h == j+2  ):
          data[i,j] = 255
      else: 
        if(h <= w or h==w or h==w+2):
          data_mask[i,j] = 255
        if(h <= w and strips%2==0 or h==w+2):
          data[i,j] = 255
  return data, data_mask

##########################################
### CREATE ACTION-DISPLAY
#########################################


def create_count_window(img_size, n=0):

  data = np.zeros((img_size,img_size), dtype=np.uint8)
  data_mask = np.zeros((img_size,img_size), dtype=np.uint8)
  line_width = int(img_size/10)+1
  if(n==1):
    draw_1(data, img_size, line_width)
  if(n==2):
    draw_2(data, img_size, line_width)
  if(n==3):
    draw_3(data, img_size, line_width)
  if(n==4):
    draw_4(data, img_size, line_width)
  if(n==5):
    draw_5(data, img_size, line_width)
  if(n==6):
    draw_6(data, img_size, line_width)
  if(n==7):
    draw_7(data, img_size, line_width)
  if(n==8):
    draw_8(data, img_size, line_width)
  if(n==9):
    draw_9(data, img_size, line_width)
  if(n==10):
    draw_P(data, img_size, line_width)  
  if(n==11):
    draw_U(data, img_size, line_width)  
  if(n==12):
    draw_T(data, img_size, line_width) 
  if(n==13):
    draw_E(data, img_size, line_width) 
  return data
  
  
  
def draw_line(img, img_size, x_start, x_end, y_start, y_end):  
  for i in range(img_size):
      for j in range(img_size):          
          if( (x_start<= i <=x_end) and (y_start<= j <=y_end)   ):
              img[j,i] = 255
              
            


def draw_line_1(img, img_size, line_width):
    draw_line(img, img_size, int(img_size/4), int(3*img_size/4)+int(line_width), 0, line_width)
    
def draw_line_2(img, img_size, line_width):
    draw_line(img, img_size, int(img_size/4), int(3*img_size/4), int(img_size/2)-int(line_width/2), int(img_size/2) + int(line_width/2))
    
def draw_line_3(img, img_size, line_width):
    draw_line(img, img_size, int(img_size/4), int(3*img_size/4) + line_width, int(img_size)-line_width, int(img_size))
    
def draw_line_4(img, img_size, line_width):
    draw_line(img, img_size, int(img_size/4), int(img_size/4)+line_width, 0, int(img_size/2))
    
def draw_line_5(img, img_size, line_width):
    draw_line(img, img_size, int(img_size/4), int(img_size/4)+line_width, int(img_size/2), int(img_size))    

def draw_line_6(img, img_size, line_width):
    draw_line(img, img_size, int(3*img_size/4), int(3*img_size/4)+line_width, 0, int(img_size/2 + line_width/2))    
    
def draw_line_7(img, img_size, line_width):
    draw_line(img, img_size, int(3*img_size/4), int(3*img_size/4)+line_width, int(img_size/2)-int(line_width/2), int(img_size))  


# Draw "T" for "Terminate"    
def draw_line_8(img, img_size, line_width):
    draw_line(img, img_size, img_size/8, int(3*img_size/4)+int(line_width), 0, line_width) 
def draw_line_9(img, img_size, line_width):
    draw_line(img, img_size, img_size/2-line_width, img_size/2+line_width, 0, img_size) 
    
    
    
    
def draw_1(img, img_size, line_width):    
    draw_line_6(img, img_size, line_width)
    draw_line_7(img, img_size, line_width)

    
    
def draw_2(img, img_size, line_width):    
    draw_line_1(img, img_size, line_width)
    draw_line_2(img, img_size, line_width)
    draw_line_3(img, img_size, line_width)
    draw_line_5(img, img_size, line_width)
    draw_line_6(img, img_size, line_width)  

def draw_3(img, img_size, line_width):    
    draw_line_1(img, img_size, line_width)
    draw_line_2(img, img_size, line_width)
    draw_line_3(img, img_size, line_width)
    draw_line_6(img, img_size, line_width)
    draw_line_7(img, img_size, line_width) 
    
def draw_4(img, img_size, line_width):    
    draw_line_2(img, img_size, line_width)
    draw_line_4(img, img_size, line_width)
    draw_line_6(img, img_size, line_width)
    draw_line_7(img, img_size, line_width)     

def draw_5(img, img_size, line_width):    
    draw_line_1(img, img_size, line_width)
    draw_line_2(img, img_size, line_width)
    draw_line_3(img, img_size, line_width)
    draw_line_4(img, img_size, line_width)
    draw_line_7(img, img_size, line_width)     

def draw_6(img, img_size, line_width):    
    draw_line_1(img, img_size, line_width)
    draw_line_2(img, img_size, line_width)
    draw_line_3(img, img_size, line_width)
    draw_line_4(img, img_size, line_width)
    draw_line_5(img, img_size, line_width)
    draw_line_7(img, img_size, line_width)     

def draw_7(img, img_size, line_width):    
    draw_line_1(img, img_size, line_width)
    draw_line_6(img, img_size, line_width)
    draw_line_7(img, img_size, line_width)     
    
    
def draw_8(img, img_size, line_width):    
    draw_line_1(img, img_size, line_width)
    draw_line_2(img, img_size, line_width)
    draw_line_3(img, img_size, line_width)
    draw_line_4(img, img_size, line_width)
    draw_line_5(img, img_size, line_width)
    draw_line_6(img, img_size, line_width)
    draw_line_7(img, img_size, line_width) 
    

def draw_9(img, img_size, line_width):    
    draw_line_1(img, img_size, line_width)
    draw_line_2(img, img_size, line_width)
    draw_line_3(img, img_size, line_width)
    draw_line_4(img, img_size, line_width)
    draw_line_6(img, img_size, line_width)
    draw_line_7(img, img_size, line_width)    

def draw_P(img, img_size, line_width):    
    draw_line_1(img, img_size, line_width)
    draw_line_2(img, img_size, line_width)
    draw_line_4(img, img_size, line_width)
    draw_line_5(img, img_size, line_width)
    draw_line_6(img, img_size, line_width)
     
def draw_U(img, img_size, line_width):    
    #draw_line_1(img, img_size, line_width)
    #draw_line_2(img, img_size, line_width)
    draw_line_3(img, img_size, line_width)
    draw_line_4(img, img_size, line_width)
    draw_line_5(img, img_size, line_width)
    draw_line_6(img, img_size, line_width)
    draw_line_7(img, img_size, line_width) 
 

def draw_T(img, img_size, line_width):    
    draw_line_8(img, img_size, line_width)
    draw_line_9(img, img_size, line_width)
    
def draw_E(img, img_size, line_width):    # End
    draw_line_1(img, img_size, line_width)
    draw_line_2(img, img_size, line_width)
    draw_line_3(img, img_size, line_width)
    draw_line_4(img, img_size, line_width)
    draw_line_5(img, img_size, line_width)
    #draw_line_6(img, img_size, line_width)
    #draw_line_7(img, img_size, line_width) 
    
    
#count_window = create_count_window(50, n=13)    
#img = Image.fromarray(count_window).resize( (400,400))   
#display(img)
