# Define some constants
KERNEL_SIZE = 3
PADDING = 1 #KERNEL_SIZE // 2

print("Loading model....??")

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
        self.test_mode = False
        
        #img_size = env.img_size 
        img_size = env.view_size
        n_count_words = env.n_words
        n_motor_actions = env.n_motor_actions
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_count_words = n_count_words
        self.task = env.task_n
        self.task_vector_length = env.task_vector_length
        
        self.draw_graphy = False
        self.network_graph_list = []


        n_actions = 4
        
        
        ## Define number of hidden neurons (attention: some are restricted)
        self.vis_representation_size = 70 #10*n_actions
        #self.lang_representation_size = 5*n_count_words+1

        #self.vis_representation_size = img_size*img_size*hidden_size
        #self.lang_representation_size = self.vis_representation_size+n_count_words+1+ self.task_vector_length
        self.memory_length = 3*(n_count_words+1)
        self.vis_lang_representation_size = self.vis_representation_size  + self.memory_length
        self.lang_representation_size = self.vis_lang_representation_size
        
        #########################
        ## Define LSTM-Gates
        ########################
        # Conv-LSTM:
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)
        # FC-LSTM:
        
        self.Gates_lang = nn.Linear((n_count_words+1)+self.memory_length+self.task_vector_length, 3*self.memory_length )
        self.Gates_lang_forget = nn.Linear((n_count_words+1)+self.memory_length+self.task_vector_length, self.memory_length )
        #self.Gates_lang_forget.bias.data.fill_(1.0)
        
        ########################
        ## Define fully connected Layers
        ##############################
        
        ## Representations
        # From flattened-2D to vis-representation        
        self.fc1 = nn.Linear(img_size*img_size*hidden_size+self.task_vector_length, self.vis_representation_size)
		# From vis-lang-task representation to lang-representation
        self.fc3_lang = nn.Linear(self.vis_lang_representation_size, self.lang_representation_size)
        
        ## Outputs: Motor-action, Verbal-action, IsDoMotoraction, IsSayWord,
        # Motor-action
        self.fc2 = nn.Linear(self.vis_lang_representation_size, n_motor_actions)
        # Word
        self.fc4_lang = nn.Linear(self.lang_representation_size, n_count_words)        
        #IsSayWord
        self.fc5_lang = nn.Linear(self.lang_representation_size, 1)
        #IsDoMotoraction
        self.fc6 = nn.Linear(self.vis_lang_representation_size, 1)
        
        
        
        
        #################
        ## Declare layers, such they are accessible from the outside
        ########################################
        self.cell = None
        self.hidden = None
        self.cell_lang = None
        self.hidden_lang = None
        self.vis_representation_layer = None
        self.output_lang_repr = None


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
               state_size = [batch_size, self.memory_length]
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
               state_size = [batch_size, self.memory_length]
               prev_state_lang = (
                   Variable(torch.zeros(state_size)).cuda(),  #.cuda()
                   Variable(torch.zeros(state_size)).cuda()  #.cuda()
               )
        #################################
        ## Conv-LSTM
        #################################
        prev_hidden, prev_cell = prev_state
        #input_ = input_*5
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
        stacked_output_and_task_vector = torch.cat((output, env_task_list), 1)
        output_repr = f.relu(self.fc1(stacked_output_and_task_vector))
        #output_repr = output
        

        #################################
        ## LSTM
        #################################
        
        prev_hidden_lang, prev_cell_lang = prev_state_lang
        input_lang_size = (self.n_count_words+1)
        stacked_inputs_lang = torch.cat((input_lang, prev_hidden_lang, env_task_list), 1).view(-1, (self.n_count_words+1)+self.memory_length+self.task_vector_length )
        
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
        if(self.test_mode == False):
            drops = nn.Dropout(p=0.2)
            output_lang = drops(output_lang)
        

 
          
		######################################
    	## Joined Vis-Lang-Task representation
        stacked_lang_vis_repr = torch.cat((output_lang, output_repr), 1)
         
        ######################################
        ## Motor action
        IsDoMotorAction = f.sigmoid(self.fc6(stacked_lang_vis_repr))        
        output = f.sigmoid(self.fc2(stacked_lang_vis_repr))        
        stacked_output = torch.cat((output, IsDoMotorAction), 1)
        
        ######################################
        ## Verbal action          
        #output_lang_repr = f.relu(self.fc3_lang(stacked_lang_vis_repr))
        output_lang_repr = stacked_lang_vis_repr
        output_lang = f.sigmoid(self.fc4_lang(output_lang_repr))
        output_isCount = f.sigmoid(self.fc5_lang(output_lang_repr))        
        stacked_output_lang = torch.cat((output_lang, output_isCount), 1)
        
        #################
        ## Define layers, such they are accessible from the outside
        ########################################
        self.cell = cell
        self.hidden = hidden
        self.cell_lang = cell_lang
        self.hidden_lang = hidden_lang
        
        self.vis_representation_layer = output_repr
        self.output_lang_repr = output_lang_repr
        
        
        if(self.draw_graphy==True):
            #print("in draw graphy")
            G=nx.DiGraph()
            self.own_G = own_G(G)
            #print(input_stacked.shape)
            #stacked_inputs_lang = torch.cat((input_lang, prev_hidden_lang, env_task_list)
            # in_gate_lang, out_gate_lang, cell_gate_lang = gates_lang.chunk(3, 1)
            # remember_gate_lang = gates_lang_forget
            output_description_list = ["End/Wait", "1", "2", "3","4","5","6","7","8","9","If-S"]
            
            input_lang_description_list = ["End/Wait", "1", "2", "3","4","5","6","7","8","9","If-S"]
            prev_hidden_lang_description_list = [""]*self.memory_length
            task_description_list = ["Touch", "Count", "Give", "Recite","Nothing","Objects", "Events","-", "-", "-", "ALL", "1", "2","3", "4","5", "6","7", "8","9"]
            
            stacked_description_list = input_lang_description_list + prev_hidden_lang_description_list + task_description_list
            
            #################
            ## Layer reservoir
            ################
            
            #add_layer(self.own_G, 0, activations = stacked_inputs_lang.tolist(),node_description_list=stacked_description_list, description_pos='middle left', layer_description='Input')            
            #add_layer(self.own_G, 1, activations = remember_gate_lang.tolist(),layer_description='Remember-gate')
            #add_layer(self.own_G, 2, activations = prev_cell_lang.tolist(),layer_description='Prev Cell', layer_operation='x')
            #add_layer(self.own_G, 3, activations = in_gate_lang.tolist(),layer_description='In-Gate', layer_operation='+')
            #add_layer(self.own_G, 4, activations = cell_gate_lang.tolist(),layer_description='Cell-Gate', layer_operation='x')
            
            #add_layer(self.own_G, 5, activations = torch.tanh(self.cell_lang).tolist(),layer_description='tanh(Cell)', layer_operation='=')
            #add_layer(self.own_G, 6, activations = out_gate_lang.tolist(),layer_description='Out gate', layer_operation='x')
            #add_layer(self.own_G, 8, activations = stacked_lang_vis_repr.tolist(), description_pos='middle right',layer_description='Stacked lang vis representation')
            
            
            
            
            # Hidden-lang + lang-output
            #add_layer(self.own_G, 0, activations = torch.tanh(self.hidden_lang).tolist(),layer_description='Language Representation') #'Hidden state - LSTM'
            #add_layer(self.own_G, 1, activations = stacked_output_lang.tolist(),node_description_list=output_description_list, description_pos='middle right',layer_description='Output')
            
            # Vis repr + lang-output
            #add_layer(self.own_G, 0, activations = output_repr.tolist(),layer_description='Visual Representation')
            #add_layer(self.own_G, 1, activations = stacked_output_lang.tolist(),node_description_list=output_description_list, description_pos='middle right',layer_description='Output')
            
            # Hidden lang + Vis repr + lang-output
            add_layer(self.own_G, 0, activations = torch.tanh(self.hidden_lang).tolist(),layer_description='Hidden state - LSTM', layer_operation='x')
            add_layer(self.own_G, 1, activations = output_repr.tolist(),layer_description='Visual Representation', layer_operation='x')
            add_layer(self.own_G, 2, activations = stacked_output_lang.tolist(),node_description_list=output_description_list, description_pos='middle right',layer_description='Output')
                
            # Lang-LSTM + Lang-output
#             add_layer(self.own_G, 0, activations = input_lang.tolist(),node_description_list=stacked_description_list, description_pos='middle left', layer_description='Input')            
#             add_layer(self.own_G, 1, activations = remember_gate_lang.tolist(),layer_description='Remember-gate')
#             add_layer(self.own_G, 2, activations = prev_cell_lang.tolist(),layer_description='Prev Cell', layer_operation='x')
#             add_layer(self.own_G, 3, activations = in_gate_lang.tolist(),layer_description='In-Gate', layer_operation='+')
#             add_layer(self.own_G, 4, activations = cell_gate_lang.tolist(),layer_description='Cell-Gate', layer_operation='x')
            
#             add_layer(self.own_G, 5, activations = torch.tanh(self.cell_lang).tolist(),layer_description='tanh(Cell)', layer_operation='=')
#             add_layer(self.own_G, 6, activations = out_gate_lang.tolist(),layer_description='Out gate', layer_operation='x')     
#             add_layer(self.own_G, 7, activations = torch.tanh(self.hidden_lang).tolist(),layer_description='tanh(hidden) language', layer_operation='x')
#             add_layer(self.own_G, 8, activations = stacked_output_lang.tolist(),node_description_list=output_description_list, description_pos='middle right',layer_description='Output')
            

            
            
            # Gate weights from Language-LSTM
            weights_i, weights_o,weights_c = np.split(self.Gates_lang.weight.data.detach().numpy()[:,:input_lang_size], 3,axis=0)
            weights_r = self.Gates_lang_forget.weight.data.detach().numpy()[:,:input_lang_size]
            # Weights from vis_lang_repr --> langoutput
            weights_vis_lang_repr_to_output_lang = self.fc4_lang.weight.data.detach()
            weights_vis_lang_repr_to_isCount = self.fc5_lang.weight.data.detach()
            
            weights_vis_lang_repr_to_whole_output_lang = torch.cat((weights_vis_lang_repr_to_output_lang, weights_vis_lang_repr_to_isCount), 0).numpy()
            
            weights_lang_repr_to_output_lang = self.fc4_lang.weight.data.detach()[:,:self.memory_length]
            weights_lang_repr_to_isCount = self.fc5_lang.weight.data.detach()[:,:self.memory_length] 
            weights_lang_repr_to_whole_output_lang = torch.cat((weights_lang_repr_to_output_lang, weights_lang_repr_to_isCount), 0).numpy()

            weights_vis_repr_to_output_lang = self.fc4_lang.weight.data.detach()[:,self.memory_length:]
            weights_vis_repr_to_isCount = self.fc5_lang.weight.data.detach()[:,self.memory_length:] 
            weights_vis_repr_to_whole_output_lang = torch.cat((weights_vis_repr_to_output_lang, weights_vis_repr_to_isCount), 0).numpy()
            #print("weights_vis_repr_to_whole_output_lang.shape: ", weights_vis_repr_to_whole_output_lang.shape)
            #print("self.fc4_lang.weight.data.detach().shape: ", self.fc4_lang.weight.data.detach().shape )
            
            
            # Hidden-lang + lang-output
            #connect_layer(self.own_G, 0, 1, weights = weights_lang_repr_to_whole_output_lang)
            # Vis repr + lang-output
            #connect_layer(self.own_G, 0, 1, weights = weights_vis_repr_to_whole_output_lang)
            # Hidden lang + Vis repr + lang-output
            connect_layer(self.own_G, 0, 2, weights = weights_lang_repr_to_whole_output_lang)
            connect_layer(self.own_G, 1, 2, weights = weights_vis_repr_to_whole_output_lang)
            # Lang-LSTM + Lang-output
#             connect_layer(self.own_G, 0, 1, weights = weights_r)
#             connect_layer(self.own_G, 0, 3, weights = weights_i)
#             connect_layer(self.own_G, 0, 4, weights = weights_c)
#             connect_layer(self.own_G, 0, 6, weights = weights_o)    
#             connect_layer(self.own_G, 7, 8, weights = weights_lang_repr_to_whole_output_lang)
            
            
            #connect_layer(self.own_G, 8, 9, weights = weights_vis_lang_repr_to_whole_output_lang)
            
            #weights_2 = self.fc1.weight.data.detach()
            #weights_3 = self.fc3.weight.data.detach()
            #weights_4 = self.fc2.weight.data.detach()
            #weights_5 = self.fc4.weight.data.detach()
            
            #lang_vis_to_output_lang_weights = torch.cat((weights_2, weights_3), 0).numpy()

            #stacked_weights = torch.cat((weights_2, weights_3,weights_4,weights_5), 0).numpy()
            #print("stacked_weights.shape: ", stacked_weights.shape)
            
            #weights_1 = self.fc1.weight.data.detach().numpy()
            #weights_2 = self.fc2.weight.data.detach().numpy()
            
            #connect_layer(self.own_G, 0, 1, weights = weights_r)
            #connect_layer(self.own_G, 0, 3, weights = weights_i)
            #connect_layer(self.own_G, 0, 4, weights = weights_c)
            #connect_layer(self.own_G, 0, 6, weights = weights_o)
            
            #connect_layer(self.own_G, 7, 8, weights = stacked_weights)
            
            self.network_graph_list.append(self.own_G)
        
        
        
        return [hidden, cell], stacked_output, [hidden_lang, cell_lang], stacked_output_lang

      
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features