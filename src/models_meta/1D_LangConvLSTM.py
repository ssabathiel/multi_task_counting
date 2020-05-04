# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

print("Loading model....")

class LangConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, env,lr):
        super().__init__()
        
        self.model_id = random.randint(1,10000) 
        self.episode = 0
        self.lr = lr
        self.result_tensory = result_tensor()
        self.time_run = 0.0
        self.model_path = ""
        
        img_size = env.img_size 
        n_count_words = env.n_words
        n_motor_actions = env.n_motor_actions
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_count_words = n_count_words
        self.task = env.task_n
        self.task_vector_length = 20 #env.task_vector_length

        n_actions = 4
        

        
        #########################
        ## Define LSTM-Gates
        ########################
        self.memory_size = 15
        # FC-LSTM:
        self.Gates = nn.Linear(env.view_size + n_count_words, 4*self.memory_size )

        
        ########################
        ## Define fully connected Layers
        ##############################
        
        ## Representations
        # From flattened-2D to vis-representation        
        self.fc1 = nn.Linear(self.memory_size, n_motor_actions)
        self.fc2 = nn.Linear(self.memory_size, n_count_words  )
        self.fc3 = nn.Linear(self.memory_size, 1 )
        self.fc4 = nn.Linear(self.memory_size, 1 )

        #print("n_motor_actions: ", n_motor_actions)
        #print("n_count_words: ", n_count_words)
        
        

    def forward(self, input_, prev_state, input_lang, prev_state_lang, env_task_list):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[1]


        # generate empty prev_state, if None is provided
        if(CUDA_bool==False):
           if prev_state is None:
               state_size = [batch_size, self.memory_size] 
               prev_state = (
                   Variable(torch.zeros(state_size)),  #.cuda()
                   Variable(torch.zeros(state_size))   #.cuda()
               )
           if(prev_state_lang is None):
               state_size = [batch_size, self.n_count_words+1]
               prev_state_lang = (
                   Variable(torch.zeros(state_size)),  #.cuda()
                   Variable(torch.zeros(state_size))  #.cuda()
               )
        else:
           if prev_state is None:
               state_size = [batch_size, self.hidden_size] + list(spatial_size)
               prev_state = (
                   Variable(torch.zeros(state_size)).cuda(),  #.cuda()
                   Variable(torch.zeros(state_size)).cuda()   #.cuda()
               )
           if(prev_state_lang is None):
               state_size = [batch_size, self.n_count_words+1]
               prev_state_lang = (
                   Variable(torch.zeros(state_size)).cuda(),  #.cuda()
                   Variable(torch.zeros(state_size)).cuda()  #.cuda()
               )
        #################################
        ## Conv-LSTM
        #################################
        prev_hidden_lang, prev_cell_lang = prev_state_lang
        prev_hidden, prev_cell = prev_state
        input_ = input_
        # data size is [batch, channel, height, width]

        stacked_inputs = torch.cat((input_, input_lang), 1)
        #print("stacked_inputs.shape: ", stacked_inputs.shape)
        gates = self.Gates(stacked_inputs)
        
        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
     
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        
        cell_gate = torch.tanh(cell_gate)
        
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        output = hidden.view(batch_size, -1 )
        
        output_action = f.sigmoid(self.fc1(output)) #output
        IsDoMotorAction = f.sigmoid(self.fc3(output))

        output_lang = f.sigmoid(self.fc2(output))
        output_isCount = f.sigmoid(self.fc4(output))        
       
        stacked_output = torch.cat((output_action, IsDoMotorAction), 1)       
        stacked_output_lang = torch.cat((output_lang, output_isCount), 1)
        
        
        self.input_vis = input_
        self.input_lang = input_lang
        self.cell = cell
        self.hidden = hidden
        self.stacked_output_lang = stacked_output_lang
        

        return [hidden, cell], stacked_output, [prev_hidden_lang, prev_cell_lang], stacked_output_lang

      
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features