
print("Load Count-Environment..")
import numpy as np


class Square():
  def __init__(self, pos, id_, n_neighbours):
    self.pos = pos
    self.id = id_
    self.n_neighbours = n_neighbours
    
    self.picked_already = False 
    self.touched_already = False 
    self.touched_count = 0  
    self.data = np.ones((1, 1), dtype=np.uint8)*255
    
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
      

class Hand():
    def __init__(self, data,data_mask, pos):
      self.data = data
      self.pos = pos
      self.data_mask = data_mask
      

class Pos():
  def __init__(self, x_, y_):
    self.x = x_
    self.y = y_

import counting_tasks  
  
class CountEnv():
    def __init__(self,task="recite_n", n_squares=1, rand_n_squares=True, save_epoch=False, display="None"):
        
        self.img_size = 4
        self.view_size = self.img_size
        if(type(n_squares)==list):
          self.n_squares_max = max(n_squares)
          self.n_squares_max_list = n_squares
        else:
          self.n_squares_max = n_squares
          self.n_squares_max_list = [self.n_squares_max]*10

        self.task = task   
        self.render = False
        self.display = display
        self.save_epoch = save_epoch

        self.count_action = False
        self.motor_action = False
        self.since_count_action = 0
        self.show_number_length = 1
        self.show_number = False
        self.last_count_number = 0
        self.rand_n_squares=rand_n_squares
        
        self.IsTripleAction = False
        self.action_motor = ""
        self.action_IsSayWord = True
        self.action_word = ""
        self.IsSayWord = False
                
        self.sample_task = False
        self.task_list = None
        self.observation_hand = []
        self.observation_square = []
        
        
        self.n_squares_wished = -1
        self.n_squares = -1
        self.n_squares_2 = -1
        self.add_n = 0
        self.quant_n_2 = -1
        
        self.max_n_wait_steps = 3
        self.SOURCE_PATH = '/content/drive/MyDrive/embodied_counting/src/'
        FONT_PATH = self.SOURCE_PATH + 'Arial.ttf'
        self.FONT_PATH = FONT_PATH
        self.fnt = ImageFont.truetype(FONT_PATH, 40)
        
        hand_path = self.SOURCE_PATH + "pics/pointing.png"
        hand_path_touch = self.SOURCE_PATH + "pics/pointing_down.png"
        sound_off_path = self.SOURCE_PATH + "pics/sound_on2.png"
        sound_on_path = self.SOURCE_PATH + "pics/sound_on2.png"
                
        self.comic_hand_size = 75
        self.sound_size = 50
        self.comic_hand = Image.open(hand_path, 'r').resize( (self.comic_hand_size,self.comic_hand_size), resample=0)
        self.comic_hand_touch = Image.open(hand_path_touch, 'r').resize( (self.comic_hand_size,self.comic_hand_size), resample=0)
        self.sound_on = Image.open(sound_on_path, 'r').resize( (self.sound_size,self.sound_size), resample=0).convert('RGBA')
        self.sound_off = Image.open(sound_off_path, 'r').resize( (self.sound_size,self.sound_size), resample=0).convert('RGBA')

        self.task_vector_size = 5 #5
        self.object_vector_size = 5 #5
        self.quant_vector_size = 10 #10

        self.n_motor_actions = 7
        self.n_words = 10        
        self.n_actions = self.n_motor_actions + self.n_words + 2

        self.task_vector_length = self.task_vector_size + self.object_vector_size + self.quant_vector_size

        self.a_strings = [["D", "U", "R", "L", "P", "Dr", "T","A", "E", "1", "2", "3","4","5","6","7","8","9","S"]]
        #self.task_node_names = ["Touch", "Count", "Give", "Recite","Nothing","ALL","1", "2","3", "4","5", "6","7", "8","9","Objects", "Events","-", "-", "-"]
        self.task_node_names = ["How \n many", "Count", "Give", "Successor","More","ALL","1", "2","3", "4","5", "6","7", "8","9","Objects", "Events","-", "-", "-"]
        
        self.action = 1
        self.action_onehot = np.array([int(i == 0) for i in range(self.n_actions)])
        self.total_reward = 0    
        self.move_dist = 1
                
        self.pick_from_00 = False
        self.pick_from_00_then_move = False
        self.pick_from_03 = False
        self.give_n_current_squares = 1
        self.readable_action_onehot = []
        
        self.print_action_onehot_aftersteps = False
        self.print_sub_tasks_after_steps = False

        self.pseudo_pos = Pos(0,0)  #here to use classes pos and square in imported counting tasks
        self.pseudo_square = Square(self.pseudo_pos, 1, 0) #(data,) pos, id_, n_neighbours


        self.reset()



        
    def reset(self):
        
        ## If multiple tasks from task_list: sample from tasks and rand_n
        sample_task_and_n_squares_if(self)
        
        if(self.task=="recite_n"):
          self.EnvTask = counting_tasks.ReciteN()
        elif(self.task=="count_all_events"):
          self.EnvTask = counting_tasks.CountAllEvents()
        elif(self.task=="count_all_objects"):
          self.EnvTask = counting_tasks.CountAllObjects()
        elif(self.task=="give_n"):
          self.EnvTask = counting_tasks.GiveN()
        elif(self.task=="touch_all_objects"):
          self.EnvTask = counting_tasks.TouchAllObjects()
        elif(self.task=="give_and_take"):
          self.EnvTask = counting_tasks.GiveAndTake()
        elif(self.task=="do_nothing"):
          self.EnvTask = counting_tasks.DoNothing()
        elif(self.task=="count_on"):
          self.EnvTask = counting_tasks.CountOn()
        elif(self.task=="recite_n_inverse"):
          self.EnvTask = counting_tasks.ReciteNInverse()
        elif(self.task=="give_give_and_take"):
          self.EnvTask = counting_tasks.GiveGiveAndTake()
        elif(self.task=="give_and_take_wo_counting"):
          self.EnvTask = counting_tasks.GiveAndTakeWOCounting()
        elif(self.task=="how_many"):
          self.EnvTask = counting_tasks.HowMany()
        elif(self.task=="more"):
          self.EnvTask = counting_tasks.More()
        elif(self.task=="successor"):
          self.EnvTask = counting_tasks.Successor()

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
        self.steps_after_given_square = 2
        self.quant_n_2 = -1
        
        self.n_wait_steps = random.randint(1,self.max_n_wait_steps)
        self.last_event_count = 0
        self.stopped_early = False
        self.one_step_after_event = False
        
        self.one_to_one_correspondence = False
        self.right_number_order = False
        self.variability = 0.0
        
        self.missed_count = False
        self.counted_none_object = False

        self.IsGrab = False
        self.IsTouch = False
        self.grabed_square = 0       
        self.reward = 0
        self.picked_once_already = False
        self.been_right_already = False
        self.is_done = False
        self.pick_from_00 = False
        self.pick_from_00_then_move = False
        self.give_n_current_squares = 1
        self.counted_word_list = [] 
        self.aimed_count_list = []
        self.counted_square_list = []
        self.given_square_id_list = []
        self.aimed_given_square_id_list = []
        self.max_time = 100
        
        
        
            
        #######
        ## Set task-specific variables
        ##################### 
        ## Task encoding: task, object and quantifier to integer
        self.EnvTask.initialize_task_vector(self)

        
        #######
        ## Initialize Env-State: Positions of squares and hand
        #####################  

        # Scene
        background = np.zeros([self.img_size,self.img_size])               
        self.background = background       
        pnt_size = 1
        
        # Hand
        pointer, pointer_mask = np.array([[255]]), np.array([[255]])
        pointer_grab, _ = np.array([[255]]), np.array([[255]])
        hand_pos_x = random.randint(0,self.img_size-1) 
        hand_pos_y = random.randint(0,self.img_size-1) 
        pos = Pos(hand_pos_x, hand_pos_y)
        hand = Hand(pointer,pointer_mask, pos)
        hand_grab = Hand(pointer_grab,pointer_mask, pos)

        # Squares with random positions
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
            self.squares = squares
        
        # Task specific initialization
        self.EnvTask.initialize_env(self)
   
        self.hand = hand        
        self.hand_nongrab = copy(hand)
        self.hand_grab = copy(hand_grab)

        self.visObs, self.observation, self.obj_belong_fct = self.constructObs()
        
        
    def constructObs(self):
            
        # Array
        self.observation = copy(self.background)
        self.observation_hand = copy(self.background)
        self.observation_square = copy(self.background)
        self.obj_belong_fct = np.zeros([self.img_size,self.img_size]).astype(int) #copy(self.background).astype(int)

        if(self.square_is_visible == True):
            for square in self.squares:     
              foreground_square = square.data  
              self.observation[square.pos.x-square.n_neighbours:square.pos.x-square.n_neighbours + square.data.shape[0], square.pos.y-square.n_neighbours:square.pos.y-square.n_neighbours + square.data.shape[1]] = foreground_square
              self.observation_square[square.pos.x-square.n_neighbours:square.pos.x-square.n_neighbours + square.data.shape[0], square.pos.y-square.n_neighbours:square.pos.y-square.n_neighbours + square.data.shape[1]] = foreground_square

              foreground_belong = (square.data/255)*square.id
              if(square.pos.x<self.img_size and square.pos.x>=0):
                 if(square.pos.y<self.img_size and square.pos.y>=0):  
                    self.obj_belong_fct[square.pos.x, square.pos.y] = square.id #foreground_belong.astype(int)
          
        #if(self.task=="give_and_take"):
        #  self.observation[0, self.img_size-1] = 150

        self.EnvTask.observation_change(self)

        observation_copy = copy(self.observation)  
        self.observationImg = Image.fromarray(observation_copy)
        
        if(self.hand_is_visible == True):
              if(self.hand.pos.x<self.img_size and self.hand.pos.x>=0):
                 if(self.hand.pos.y<self.img_size and self.hand.pos.y>=0):
                    observation_copy[self.hand.pos.x, self.hand.pos.y] = 127        
                    self.observation_hand[self.hand.pos.x, self.hand.pos.y] = 255

        if(self.render == True or self.display != "None"):
            self.observationImg = self.render_env()
        
        return self.observationImg, self.observation, self.obj_belong_fct
      
    def solve_task(self):
        self.EnvTask.solve_task(self)


    def render_env(self):
        ######################
        ## Render Environment   
        self.observationImg = self.observationImg.resize( (400,400), resample=0).convert('RGB')

        if(self.IsTouch == False and self.IsGrab==False):
          handy_img = self.comic_hand
        else:
          handy_img = self.comic_hand_touch

        
        ###########
        # ADD HAND, TASK-VECTOR IMG, SOUND-IMG
        #################
        # HAND
        offset = (self.hand.pos.y*100+25, self.hand.pos.x*100+25)
        self.observationImg.paste(handy_img, offset, handy_img)
        self.observationImg = ImageOps.expand(self.observationImg,border=7,fill='black')
        # TASK
        task_img = get_task_img(self)
        self.observationImg = get_concat_v_env(self.observationImg, task_img).convert('RGB')

        # SOUND
        if(self.IsSayWord): #self.IsSayWord self.show_number
          sound_img = self.sound_on
          stringy = self.action_word #Action[self.action_word]
        else:
          sound_img = self.sound_off
          stringy = ""
        self.observationImg = get_concat_v_env(sound_img, self.observationImg)
        d = ImageDraw.Draw(self.observationImg)          
        d.text((70,0), stringy, fill=(0,0,0), font=self.fnt)
        
        
        #### Count window
        if(self.show_number == True and self.show_number == False):
            count_window_size = 8
            intensity_factor = 1 # np.sin(self.since_count_action/self.show_number_length*np.pi)
            count_window = create_count_window(count_window_size, n=self.last_count_number )*intensity_factor
            count_window = Image.fromarray(count_window).resize((50,50), resample=0).convert('RGB')
            bord_dist = int(self.img_size/20)       
            bg_w, bg_h = self.observationImg.size
            img_w, img_h = count_window.size
            offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
            if(self.task == "count_all_events"):
              offset = ((bg_w - img_w) // 12, (bg_h - img_h) // 12)
            self.observationImg.paste(count_window, offset)

        return self.observationImg



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
            action_onehot[self.n_motor_actions] = 1
                
        if(action_IsSayWord):
            action_onehot[Action_inv[action_word]+1] = 1
            action_onehot[-1] = 1
       
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
        
        self.action_onehot = self.triple_action_one_hot(self.action_motor, self.IsDoMotorAction, self.action_word, self.IsSayWord)        
        self.IsTripleAction = False
                
        if(self.print_action_onehot_aftersteps):
          self.print_actions()

        if(self.save_epoch):
            img, action = envImageAndActionToPytorchFormat(self)
            action_string = self.readable_actionstring()
            curr_exp = {'img': img, 'action': action, 'dem_img': self.observationImg, 'action_string': action_string}
            self.epoch.append(curr_exp)
            
        self.visObs = self.constructObs()
        self.IsTouch = False
        
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
      
      
      self.reward = 0
      
      move_dist=self.move_dist
      
      if(type(action)==int):
        self.action = Action[action]
      else:
        self.action = action
      
      if(self.IsSayWord):
        self.action_word = self.action
      
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
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = 10     
      if(Action_inv[self.action]==5):
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = 11  
          
      if(Action_inv[self.action]==6):
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = 12 
          
      if(Action_inv[self.action]==7):
          self.since_count_action = 0
          self.show_number = True
          self.last_count_number = 13 
          
      if(self.since_count_action < self.show_number_length):
          self.since_count_action += 1
      else: 
          self.show_number = False
      
      
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
        if(self.obj_belong_fct[self.hand.pos.x, self.hand.pos.y] != 0):
          self.grabed_square = self.obj_belong_fct[self.hand.pos.x, self.hand.pos.y]
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
            self.is_done = True            
        #self.IsGrab = False  
        self.hand.data = copy(self.hand_nongrab.data)
       
      elif(self.action=="touch"):     
          self.action_onehot = np.array([int(i == 6) for i in range(self.n_actions)])        
          if(self.obj_belong_fct[self.hand.pos.x + int(self.hand.data.shape[0]/2), self.hand.pos.y + int(self.hand.data.shape[1]/2)] != 0):
              self.touched_square = self.obj_belong_fct[self.hand.pos.x + int(self.hand.data.shape[0]/2), self.hand.pos.y + int(self.hand.data.shape[1]/2)]
              self.IsTouch = True
              self.hand.data = copy(self.hand_grab.data)
              self.squares[self.touched_square-1].touched_already = True
              self.squares[self.touched_square-1].touched_count += 1
          if(self.obj_belong_fct[self.hand.pos.x + int(self.hand.data.shape[0]/2), self.hand.pos.y + int(self.hand.data.shape[1]/2)] == 0):   
              self.counted_none_object = True
              
      elif(self.action=="stop"):     
        self.action_onehot = np.array([int(i == 7) for i in range(self.n_actions)])
   


        
      ######################
      ## Create action_onehot
      ######################+
      
      if(self.IsTripleAction == False):
        self.IsDoMotorAction = False
        self.IsSayWord = False
      
      # Find out if action is motor action 
      if(Action_inv[self.action]<self.n_motor_actions):
          self.IsDoMotorAction = True
                 
      #Find out if action is saying word
      if(Action_inv[self.action]>self.n_motor_actions-1):    
          self.IsSayWord = True
                    
      self.action_onehot = self.triple_action_one_hot(self.action, self.IsDoMotorAction, self.action, self.IsSayWord)
      if(self.print_action_onehot_aftersteps and self.IsTripleAction==False):
          self.print_actions()
      
      
   
      ##################
      ## UPDATE ENVIRONMENT
      #####################
      self.EnvTask.update_variables(self)
        
        
        
      ###########################
      ### SET END CONDITIONS
      ###########################
      self.EnvTask.check_solved_condition(self)
      #print("self.ended: ", self.ended)

      ###########################
      ### SAVE-EPOCH
      ###########################
      if(self.IsTripleAction == False):
          if(self.save_epoch):              
              img, action = envImageAndActionToPytorchFormat(self)
              object_features = get_object_features(self)
              action_string = self.readable_actionstring()
              curr_exp = {'img': img, 'action': action, 'dem_img': self.observationImg, 'action_string': action_string}
              self.epoch.append(curr_exp)              
          self.visObs = self.constructObs()
          
      ################
      ## Construct Observation and Render-If
      ####################################
      self.visObs = self.constructObs()
      if(self.display != "None"):
          update_display(self.observationImg.convert('RGB'), display_id = "game")
          time.sleep(delay_time)
    
    

    
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



def sample_task_and_n_squares_if(self):
    if(self.sample_task==True):
        self.task = random.sample(self.task_list,1)[0]  
    
    if(self.task_list is None):
      self.task_n = 0
    else:
      self.task_n = self.task_list.index(self.task)

    if(self.rand_n_squares==True):
        self.n_squares = random.randint(1,self.n_squares_max_list[self.task_n])
        self.n_squares_2 = random.randint(1,self.n_squares_max_list[self.task_n])
        self.add_n = random.randint(1,self.n_squares_max_list[self.task_n] - self.n_squares + 1)
    else:
        self.n_squares = self.n_squares_wished

    #if(self.rand_n_squares):
        #print("task: ", self.task) 
        #print("self.n_squares: ", self.n_squares)
        #print("task_n: ", self.task_n) 
        #print("self.n_squares_max_list[self.task_n] ", self.n_squares_max_list[self.task_n])
        #print("-----------------")

    #if(self.n_squares_wished >= 0):
    #    self.n_squares = self.n_squares_wished
    
    #self.task_n = 0





def create_pointer(width, height, pick=False):
    
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
 
    

    
    
    
def get_concat_h_env(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_env(im1, im2):
    dst = Image.new('RGB', (im2.width, im1.height + im2.height), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
  
#count_window = create_count_window(50, n=13)    
#img = Image.fromarray(count_window).resize( (400,400))   
#display(img)





def get_task_img(env):
    task_vector = create_task_vector(env)
    task_vector = task_vector.reshape(1, env.task_vector_length)
    task_vector = task_vector.detach().numpy().reshape(env.task_vector_length)


    double_task_vector = np.stack( (task_vector,task_vector))
    img_width = env.task_vector_length*45
    img_height = img_width//20
    img = Image.fromarray(double_task_vector*255).resize( (img_width,img_height),resample=Image.NEAREST).convert('RGB') #.resize( (img_width,img_hight)).convert('RGB')

    # Draw boundary around total vector and inbetween elements
    boundary_color = (0,0,255)
    boundary_width = 4

    drawy = ImageDraw.Draw(img)
    for i in range(env.task_vector_length):
      drawy.line((i*img_height - boundary_width//2 ,0, i*img_height- boundary_width//2,img_height), fill=boundary_color, width = boundary_width )

    border_width = 5
    img = ImageOps.expand(img, border=border_width, fill = boundary_color)

    text_img = Image.new('RGB', (img.width, img.height), color='white')
    draw = ImageDraw.Draw(text_img)
    FONT_PATH = env.FONT_PATH
    task_font = ImageFont.truetype(FONT_PATH, 24)
    node_font = ImageFont.truetype(FONT_PATH, 14)

    # Annotate names of nodes
    #node_names_list = ["Touch", "Recite", "Count", "Give","Nothing","1", "2","3", "4","5", "6","7", "8","9", "ALL","Objects", "Events","-", "-", "-"]
    node_names_list = env.task_node_names
    for w_i in range(len(node_names_list)):
      wordy = node_names_list[w_i]
      draw.text((border_width + w_i*img_height + img_height//2 - 3*len(wordy), 40),wordy,(0,0,0), font=node_font)

    task_img = get_concat_v(text_img, img, distance=0)


    return task_img



def network_array_to_image_new(env, layer_description=None, img_width=45):   #not used right now
      
      node_names_list = env.task_node_names
      array_ = create_task_vector(env)
      array_size = array_.size

      array_ = array_.reshape(array_size)

      '''
      for i in range(task_vector.size):
        if(i%2==0):
          task_vector[i] = 0.5
      '''
      double_array_ = np.stack( (array_,array_))

      img_width = array_size*img_width
      img_height = img_width//array_size
      img = Image.fromarray(double_array_*255).resize( (img_width,img_height),resample=Image.NEAREST).convert('RGB') 

      boundary_color = (0,0,255)
      boundary_width = 4

      drawy = ImageDraw.Draw(img)
      for i in range(array_size):
        drawy.line((i*img_height - boundary_width//2 ,0, i*img_height- boundary_width//2,img_height), fill=boundary_color, width = boundary_width )

      border_width = 5
      img = ImageOps.expand(img, border=border_width, fill = boundary_color)

      text_img = Image.new('RGB', (img.width, img.height), color='white')
      draw = ImageDraw.Draw(text_img)

      task_font = ImageFont.truetype("/content/drive/My Drive/Embodied_counting/src/Arial.ttf", 24)
      node_font = ImageFont.truetype("/content/drive/My Drive/Embodied_counting/src/Arial.ttf", 12)


      #node_names_list = ["Touch", "Recite", "Count", "Give","Nothing","1", "2","3", "4","5", "6","7", "8","9", "ALL","Objects", "Events","-", "-", "-"]
      if(node_names_list is not None):
          for w_i in range(len(node_names_list)):
            wordy = node_names_list[w_i]
            draw.text((border_width + w_i*img_height + img_height//2 - 3*len(wordy), 40),wordy,(0,0,0), font=node_font)


      description_img = Image.new('RGB', (img.width, img.height), color='white')
      draw_description = ImageDraw.Draw(description_img)
      if(layer_description is not None):
          wordy = layer_description
          draw_description.text((img_width//2 - 3*len(wordy), 0),wordy,(0,0,0),font=task_font)

      array_img = get_concat_v(text_img, img, distance=0)
      array_img = get_concat_v(array_img, description_img, distance=0)

      return array_img










