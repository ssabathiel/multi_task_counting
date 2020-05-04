##################################
## Automatic solving algorithm
##################################


print("Loading Automatic Solving Algorithms..")

import math

delay_time = 0.5

def find_next_object(env, source="all"):
  
    d_min = 1e8
    x_min = 1e8
    y_min = 1e8
    
    square_n = -1
    n = - 1
    
    isFoundAny = False
    
    ind_list_min_dist = []
    ind_list_min_dist_min_x = []
    ind_list_min_dist_min_x_min_y = []
    
    for square in env.squares:
        
        square_criterion = True
        if(env.task=="give_n"):
            square_criterion = square.picked_already
        else:
            square_criterion = square.touched_already
        
        #n += 1  
        n = square.id - 1
        
        #print("n ", n)
        criterion = True
        if(source == "left"):
            criterion = False
            if( (square.pos.y < int(env.img_size/2.0)) and (square_criterion==False) ):
                criterion = True
           
        #print("square ", n, " picked already?: ", square.picked_already)  
        if(criterion):
            d_curr = (env.hand.pos.x - square.pos.x)**2 + (env.hand.pos.y - square.pos.y)**2
            if((d_curr <= d_min) and (not square_criterion) ):
                #print("d_curr<d_min and not picked already ", n)
                
                if(d_curr==d_min):
                  if(square.pos.x <= x_min and square.pos.y <= y_min):
                    square_n = n
                    x_min = square.pos.x
                    y_min = square.pos.y
                    isFoundAny = True
                if(d_curr<d_min):
                    square_n = n
                    x_min = square.pos.x
                    y_min = square.pos.y
                    isFoundAny = True 
                  
                d_min = d_curr

                
    return square_n, isFoundAny
  



def move_to_square(env, n):
    
      reached = False
      
      squ_x = env.squares[n].pos.x
      squ_y = env.squares[n].pos.y
      
      hand_x = env.hand.pos.x 
      hand_y = env.hand.pos.y 

      d_x = hand_x - squ_x
      d_y = hand_y - squ_y
      
      while(not reached):

          action = "none"
          if(abs(d_x) > abs(d_y)):

              if(d_x > 0):
                  env.update("up")
                  action = "up"
              else:  
                  env.update("down")
                  action = "down"
          else:
          
              if(d_y > 0):
                  env.update("left")
                  action = "left"
              else:  
                  env.update("right")
                  action = "right"
                  
          if(env.display != "None"):
              update_display(env.observationImg.convert('RGB'), display_id = "game")
              time.sleep(delay_time)
         
          
          hand_x = env.hand.pos.x 
          hand_y = env.hand.pos.y
          
          d_x = hand_x - squ_x
          d_y = hand_y - squ_y
          

          if(hand_x==squ_x and hand_y==squ_y):
              reached = True
              
      
    




def move_to_target(env):
  
    reached = False
    while(not reached):
        
        env.update("right")
        if(env.hand.pos.y >= int(0.75*env.img_size)):
            
            reached = True

        if(env.display != "None"):
            update_display(env.observationImg.convert('RGB'), display_id = "game")
            time.sleep(delay_time)





def pick_next_object(env):
    
    n, isFoundAny = find_next_object(env)
    move_to_square(env, n)
    env.update("pick")
    if(env.display != "None"):
         update_display(env.observationImg.convert('RGB'), display_id = "game")
         time.sleep(delay_time)


def move_square_from_source_to_target(env, n_sofar = 0):
    
    n, isFoundAny = find_next_object(env, source = "left" )
    
    
    if(isFoundAny):
        
        move_to_square(env, n)
        env.update("pick")
        if(env.display != "None"):
             update_display(env.observationImg.convert('RGB'), display_id = "game")
             time.sleep(delay_time)
        move_to_target(env)
        
        if(env.task == "give_n"):
          #env.triple_update("release", True, str(n_sofar))
          
          if(n_sofar==env.n_squares):
              env.triple_update("release", True, "stop", False)
              if(env.display != "None"):
                 update_display(env.observationImg_between.convert('RGB'), display_id = "game")
                 time.sleep(delay_time)
          else:
              env.update("release")
          if(env.display != "None"):
             update_display(env.observationImg.convert('RGB'), display_id = "game")
             time.sleep(delay_time)
          #env.update(str(n_sofar))
        else: 
           env.update("release")          
        if(env.display != "None"):
             update_display(env.observationImg.convert('RGB'), display_id = "game")
             time.sleep(delay_time)
              
           
    
    return (not isFoundAny)




def touch_all_objects(env):
  
  isFoundAny = True
  
  
  while(isFoundAny):
    n, isFoundAny = find_next_object(env, source = "all" )
    if(isFoundAny):
        
        move_to_square(env, n)
        if(env.display != "None"):
            update_display(env.observationImg.convert('RGB'), display_id = "game")
            time.sleep(delay_time)
        env.triple_update("touch", True, "1", False)
        if(env.display != "None"):
             update_display(env.observationImg.convert('RGB'), display_id = "game")
             time.sleep(delay_time)


              
def count_all_objects(env):
  
  isFoundAny = True
  n_sofar=0
  
  
  while(isFoundAny):
    n, isFoundAny = find_next_object(env, source = "all" )
    n_sofar += 1
    if(isFoundAny):
        
        move_to_square(env, n)
        
        
        env.triple_update("touch", True, str(n_sofar), True)
        
        
        
        if(env.display != "None"):
             update_display(env.observationImg_between.convert('RGB'), display_id = "game")
             time.sleep(delay_time)
        if(env.display != "None"):
             update_display(env.observationImg.convert('RGB'), display_id = "game")
             time.sleep(delay_time)
    



def move_all_squares_from_source_to_target(env):
  
    done = False
    
    while(not done):
        done = move_square_from_source_to_target(env)


def give_n(env):
  
    done = False
    n = env.n_squares
    given_squares_so_far = 0
    
    while(given_squares_so_far != n):
        given_squares_so_far += 1
        done = move_square_from_source_to_target(env, given_squares_so_far)
        
    env.update("stop") 

    if(env.display != "None"):
        update_display(env.observationImg.convert('RGB'), display_id = "game")
        time.sleep(delay_time)

    
    
def count_all_events(env):
  
    done = False
    n = env.n_squares
    n_sofar = 0
    
    while(n_sofar < n):
        
        event_there = copy(env.event_there)
        
        if(event_there):
            n_sofar += 1
            env.triple_update("touch", False, str(n_sofar), True)
            if(env.display != "None"):
              update_display(env.observationImg.convert('RGB'), display_id = "game")
              time.sleep(1.5*delay_time) 
        else: 
            env.triple_update("touch", False, "stop", True) 
            if(env.display != "None"):
              update_display(env.observationImg.convert('RGB'), display_id = "game")
              time.sleep(1.5*delay_time)

def recite_n(env):
  
    done = False
    n = env.n_squares
    n_sofar = 0
    
    while(n_sofar < n):
        
        n_sofar += 1
        env.triple_update("touch", False, str(n_sofar), True)
               
        if(env.display != "None"):
          update_display(env.observationImg.convert('RGB'), display_id = "game")
          time.sleep(1.5*delay_time) 
        
    env.update("stop")  
    #env.triple_update("touch", False, "stop", True)
    if(env.display != "None"):
      update_display(env.observationImg.convert('RGB'), display_id = "game")
      time.sleep(1.5*delay_time)        

        
        
        




