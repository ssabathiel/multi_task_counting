print("Loading Automatic Solving Algorithms..?")

import math

delay_time = 0.5

def find_next_object(env):
  
    d_min = 1e8
    square_n = -1
    n = - 1
    
    isFoundAny = False
    
    for square in env.squares:
        n+=1
        square_criterion = True
        if(env.task=="give_n"):
            square_criterion = not square.picked_already
        else:
            square_criterion = not square.touched_already

        if(square_criterion):
          d_curr = (env.hand.pos.x - square.pos.x)**2 + (env.hand.pos.y - square.pos.y)**2
          if((d_curr < d_min) and (square_criterion) ):
            d_min = d_curr
            square_n = n
            isFoundAny = True        
                
    return square_n, isFoundAny
  

def get_square_id_at_pos(env, pos):
  
    d_min = 1e8
    square_n = -1
    n = - 1
    
    isFoundAny = False
    
    for square in env.squares:
        
        square_criterion = True
        if(env.task=="give_n"):
            square_criterion = square.picked_already
        else:
            square_criterion = square.touched_already
        
        n += 1        

        if( (square.pos.y == pos[1]) and (square.pos.x == pos[0]) ):
          square_n = n
          isFoundAny = True
                
    return square_n, isFoundAny
  

def move_to_square(env, n):
      
      squ_x = env.squares[n].pos.x
      squ_y = env.squares[n].pos.y
      
      move_to_pos = [squ_x, squ_y]      
      move_to(env, move_to_pos)
              
      
    
def move_to(env, to_pos):
    
      reached = False
      
      to_pos_x = to_pos[0]
      to_pos_y = to_pos[1]
      
      hand_x = env.hand.pos.x 
      hand_y = env.hand.pos.y 

      d_x = hand_x - to_pos_x
      d_y = hand_y - to_pos_y
      
      while(not reached):
          if(abs(d_x) > abs(d_y)):

              if(d_x > 0):
                  env.update("up")
              else:  
                  env.update("down")
          else:
          
              if(d_y > 0):
                  env.update("left")
              else:  
                  env.update("right")
  
          hand_x = env.hand.pos.x 
          hand_y = env.hand.pos.y

          d_x = hand_x - to_pos_x
          d_y = hand_y - to_pos_y
          
          if(hand_x==to_pos_x and hand_y==to_pos_y):
              reached = True


def pick_next_object(env):
    
    n, isFoundAny = find_next_object(env)
    move_to_square(env, n)
    env.update("pick")



def move_square_from_to(env, from_pos=[0,0], to_pos=[0,0]):
    
    #n, isFoundAny = get_square_id_at_pos(env, from_pos)
        
    #if(isFoundAny):        
        #move_to_square(env, n)
    move_to(env, from_pos)
    env.update("pick")
    move_to(env, to_pos)
    env.update("release")          

    #return (not isFoundAny)


    