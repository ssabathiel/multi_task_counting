# Define some constants
KERNEL_SIZE = 1
PADDING = KERNEL_SIZE // 2

print("Loading model..!")

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
        
        
        ## Define number of hidden neurons (attention: some are restricted)
        self.vis_representation_size = 10*n_actions
        self.lang_representation_size = n_count_words+5  #5*n_count_words+1

        #self.vis_representation_size = img_size*img_size*hidden_size
        #self.lang_representation_size = self.vis_representation_size+n_count_words+1+ self.task_vector_length
        
        self.vis_lang_representation_size = self.vis_representation_size  + self.task_vector_length
        
        #########################
        ## Define LSTM-Gates
        ########################
        # Conv-LSTM:
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)
        # FC-LSTM:
        self.Gates_lang = nn.Linear(2*(n_count_words+1)+self.task_vector_length + 1 + 1, 3*(n_count_words+1) )
        self.Gates_lang_forget = nn.Linear(2*(n_count_words+1)+self.task_vector_length + 1 + 1, (n_count_words+1) )
        #self.Gates_lang = self.Gates_lang = nn.Conv2d(1,4,1)
        self.Gates_lang_forget.bias.data.fill_(1.0)
        
        ########################
        ## Define fully connected Layers
        ##############################
        
        ## Representations
        # From flattened-2D to vis-representation        
        self.fc1 = nn.Linear(img_size*img_size*hidden_size, self.vis_representation_size)
		# From vis-lang-task representation to lang-representation
        self.fc3_lang = nn.Linear(n_count_words+1 + self.task_vector_length, self.lang_representation_size)
        
        ## Outputs: Motor-action, Verbal-action, IsDoMotoraction, IsSayWord,
        # Motor-action
        self.fc2 = nn.Linear(self.vis_lang_representation_size, n_motor_actions)
        # Word
        self.fc4_lang = nn.Linear(self.lang_representation_size, n_count_words) 
        #self.fc4_lang = nn.Linear(n_count_words+1 + self.task_vector_length, n_count_words)        
        #IsSayWord
        self.fc5_lang = nn.Linear(self.lang_representation_size, 1)
        #IsDoMotoraction
        self.fc6 = nn.Linear(self.vis_lang_representation_size, 1)
        
        self.fc_isCount = nn.Linear(self.vis_lang_representation_size,1)
        self.fc_context = nn.Linear(self.vis_lang_representation_size,1)

    def forward(self, input_, prev_state, input_lang, prev_state_lang, env_task_list):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if(CUDA_bool==False):
           if prev_state is None:
               state_size = [batch_size, self.hidden_size] + list(spatial_size)
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
        prev_hidden, prev_cell = prev_state
        input_ = input_*5
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
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
        output_repr = f.relu(self.fc1(output))
        
        #output_isCount = f.sigmoid(self.fc5_lang(output_lang_repr))  
        #output_repr = output
        
		######################################
    	## Joined Vis-Lang-Task representation
        stacked_lang_vis_repr = torch.cat((output_repr,env_task_list), 1)
         
        ######################################
        ## Motor action
        IsDoMotorAction = f.sigmoid(self.fc6(stacked_lang_vis_repr))        
        output = f.sigmoid(self.fc2(stacked_lang_vis_repr))        
        stacked_output = torch.cat((output, IsDoMotorAction), 1)
        
        output_isCount = f.sigmoid(self.fc_isCount(stacked_lang_vis_repr))
        word_context = f.relu(self.fc_context(stacked_lang_vis_repr) )  #should late give a hint to the language channel whether it should be counting word or stop...
        
        
        #################################
        ## LSTM
        #################################
        prev_hidden_lang, prev_cell_lang = prev_state_lang
        # output_isCount = 
        # word_context = 
        # .cat(...., fc_new_1(f.relu(fc_new( input_.view(batch_size, -1) ) ))
        stacked_inputs_lang = torch.cat((input_lang, prev_hidden_lang, env_task_list, output_isCount, word_context), 1).view(-1, 2*(self.n_count_words+1)+self.task_vector_length + 1 + 1)
        
        #gates_lang = self.Gates_lang(stacked_inputs_lang)        
        #in_gate_lang, remember_gate_lang, out_gate_lang, cell_gate_lang = gates_lang.chunk(4, 1)
        gates_lang = self.Gates_lang(stacked_inputs_lang) 
        gates_lang_forget = self.Gates_lang_forget(stacked_inputs_lang)
        in_gate_lang, out_gate_lang, cell_gate_lang = gates_lang.chunk(3, 1)
        remember_gate_lang = gates_lang_forget
        
        cell_gate_lang = torch.tanh(cell_gate_lang)
        in_gate_lang = torch.sigmoid(in_gate_lang)
        remember_gate_lang = torch.sigmoid(remember_gate_lang)
        out_gate_lang = torch.sigmoid(out_gate_lang)        
        
        
        cell_lang = (remember_gate_lang * prev_cell_lang) + (in_gate_lang * cell_gate_lang)
        hidden_lang = out_gate_lang * torch.tanh(cell_lang)
        output_lang = hidden_lang.view(batch_size, -1 )
        #output_lang = cell_lang.view(batch_size, -1 )
        stacked_output_lang_task_list = torch.cat((output_lang,env_task_list), 1)
        

 
          
  
        
        ######################################
        ## Verbal action          
        output_lang_repr = f.relu(self.fc3_lang(stacked_output_lang_task_list))
        #output_lang_repr = f.relu(self.fc3_lang(output_lang))  #stacked_output_lang_task_list
        #output_lang_repr = f.relu(self.fc3_lang(stacked_lang_vis_repr))
        #output_lang_repr = stacked_lang_vis_repr
        output_lang = f.sigmoid(self.fc4_lang(output_lang_repr))
        #output_lang = f.sigmoid(self.fc4_lang(stacked_output_lang_task_list))
        #output_isCount = f.sigmoid(self.fc5_lang(output_lang_repr))        
        stacked_output_lang = torch.cat((output_lang, output_isCount), 1)
        
        
        ## Make internal representation accessible from outside the class
        # LSTM
        self.prev_state_lang = prev_state_lang
        self.prev_state = prev_state
        self.cell = cell
        self.hidden = hidden
        self.cell_lang = cell_lang
        self.hidden_lang = hidden_lang
        
        # After
        self.vis_repr = output_repr
        self.lang_repr = output_lang_repr
        self.stacked_lang_vis_repr = stacked_lang_vis_repr
        
        

        return [hidden, cell], stacked_output, [hidden_lang, cell_lang], stacked_output_lang

      
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features