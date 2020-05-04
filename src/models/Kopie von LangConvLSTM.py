# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

print("Loading model...")

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
        
        img_size = env.img_size 
        n_count_words = env.n_words
        n_motor_actions = env.n_motor_actions
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_count_words = n_count_words
        self.task = env.task_n
        self.task_vector_length = 20 #env.task_vector_length

        n_actions = 4
        
        
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

        self.fc1 = nn.Linear(img_size*img_size*hidden_size, 10*n_actions)
        self.fc2 = nn.Linear(10*n_actions+n_count_words+1 + self.task_vector_length, n_motor_actions)
        
        self.Gates_lang = nn.Linear(2*(n_count_words+1)+self.task_vector_length, 4*(n_count_words+1) )
        #self.fc3_lang = nn.Linear(n_count_words+1, n_count_words+1)
        self.fc3_lang = nn.Linear(10*n_actions+n_count_words+1+ self.task_vector_length, n_count_words+1)
        self.fc4_lang = nn.Linear(n_count_words+1, n_count_words)
        
        #IsSayWord
        self.fc5_lang = nn.Linear(n_count_words+1, 1)
        #IsDoMotoraction
        self.fc6 = nn.Linear(10*n_actions+n_count_words+1 + self.task_vector_length, 1)

    def forward(self, input_, prev_state, input_lang, prev_state_lang, env_task_list):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        #task_encode_length = env_task_list[0].size()[0]

        # generate empty prev_state, if None is provided
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
                
            
        prev_hidden, prev_cell = prev_state
        prev_hidden_lang, prev_cell_lang = prev_state_lang

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        
        stacked_inputs_lang = torch.cat((input_lang, prev_hidden_lang, env_task_list), 1).view(-1, 2*(self.n_count_words+1)+self.task_vector_length )   #### --> does this work out with 1d arrays? --> check
        

        gates_lang = self.Gates_lang(stacked_inputs_lang)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        
        in_gate_lang, remember_gate_lang, out_gate_lang, cell_gate_lang = gates_lang.chunk(4, 1)  #### --> does this work out with 1d arrays? --> check

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        
        in_gate_lang = torch.sigmoid(in_gate_lang)
        remember_gate_lang = torch.sigmoid(remember_gate_lang)
        out_gate_lang = torch.sigmoid(out_gate_lang)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        cell_gate_lang = torch.tanh(cell_gate_lang)
        


        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        output = hidden.view(batch_size, -1 )
        output_repr = f.relu(self.fc1(output)) 
        
        
        
                
        cell_lang = (remember_gate_lang * prev_cell_lang) + (in_gate_lang * cell_gate_lang)
        hidden_lang = out_gate_lang * torch.tanh(cell_lang)
        output_lang = hidden_lang.view(batch_size, -1 )
                
        stacked_lang_vis_repr = torch.cat((output_lang, output_repr,env_task_list), 1)
                
        IsDoMotorAction = f.sigmoid(self.fc6(stacked_lang_vis_repr))
        #output = f.softmax(self.fc2(stacked_lang_vis_repr))
        output = f.sigmoid(self.fc2(stacked_lang_vis_repr))
        
        stacked_output = torch.cat((output, IsDoMotorAction), 1)
        
        #output = f.softmax(self.fc2(output_repr))
       
        #output_lang_repr = f.relu(self.fc3_lang(output_lang))   
        output_lang_repr = f.relu(self.fc3_lang(stacked_lang_vis_repr))  
        #output_lang = f.softmax(self.fc4_lang(output_lang_repr))
        output_lang = f.sigmoid(self.fc4_lang(output_lang_repr))
        output_isCount = f.sigmoid(self.fc5_lang(output_lang_repr))
        
        stacked_output_lang = torch.cat((output_lang, output_isCount), 1)
        
        #print("cell.size() ", cell.size())
        #print("hidden.size() ", hidden.size())

        return [hidden, cell], stacked_output, [hidden_lang, cell_lang], stacked_output_lang
        #return hidden, cell
      
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features