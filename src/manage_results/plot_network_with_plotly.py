###
# Fully functioning: but without node color

#from copy import copy

init_node_size = 20 #20
active_node_size = 30 #30
node_description_size = 14

init_node_color = '#a3a7e4'
button_size = 40
button_text_size = 15


active_node_color = '#a3a7e4'

init_weight_color = 'rgba(0,0,0,0.10)'
active_weight_color_positive = 'rgb(0,0,255)'
active_weight_color_negative = 'rgb(255,0,0)'
#n_time_steps = len(graph_list)


class plotly_interaction():
    def __init__(self, graph_list, fig, scatter):
        self.fig = fig
        self.scatter = scatter
        self.graph_list = graph_list
        n_time_steps = len(graph_list)
        
        self.list_of_pure_color_lists = get_list_of_pure_color_lists(graph_list)

        self.prev_btn = widgets.Button(description='t -')
        self.prev_btn.on_click(self.prev_btn_event_handler)

        self.next_btn = widgets.Button(description='t +')
        self.next_btn.on_click(self.next_btn_event_handler)

        self.time_slider = widgets.IntSlider( value=0,min=0,max=n_time_steps-1)
        self.time_slider.observe(self.on_value_change)
        
        self.time_control_fig = HBox([self.prev_btn,self.time_slider, self.next_btn])
    
    def update_point(self, trace, points, selector):
        c = list(self.scatter.marker.color)
        s = list(self.scatter.marker.size)
        turn_down_again = False
        for i in points.point_inds:

            if(s[i]==init_node_size):
                s[i] = active_node_size
                #c[i] = active_node_color
                turn_down_again = False
            else:
                s[i] = init_node_size
                #c[i] = init_node_color
                turn_down_again = True
                #print("i: ", i)
            with self.fig.batch_update():
                self.scatter.marker.color = c
                self.scatter.marker.size = s
            #Modify attached edges to clicked node i
            #find out attached edges:
            g = self.graph_list[0].G
            x_y_weight_list = list(g.edges.data('weight'))
            sub_indices = [j for j,edge in enumerate(x_y_weight_list) if (edge[0] == i or edge[1] == i)]
            #for k in sub_indices:
            #    fig.data[k].line = {'color': 'blue', 'width': 5}
            if(turn_down_again):
                colory = init_weight_color
            else:
                colory = active_weight_color_positive


            #fig_data_list = copy( list(orig_fig_data)[:] )
            fig_data_list = list(self.fig.data)
             ## formally not here!!!!!!!!!!!!!!!!!

            for k in sub_indices[::-1]: 
                # choose color of displayed weight-line: positive--> blue, negative-->red
                if(turn_down_again==False):
                    if(list(g.edges.data('weight'))[k][2]>0):
                        colory = active_weight_color_positive
                    else:
                        colory = active_weight_color_negative
                else:
                    colory = init_weight_color
                #go through reversed list of sub_indices,such that when pushed to back, the ones before keep their right index
                fig_data_list[k].line = {'color': colory, 'width': abs(list(g.edges.data('weight'))[k][2]) }
                # put edge to the back of all traces, not behind nodes (last element of fig.data):
                fig_data_list.insert(-2, fig_data_list.pop(k))           
            #fig.data = fig_data_list



            #fig.data = [fig_data_list[i] for i in sub_indices].append(node_trace)
            #fig.data.append(node_trace)

            '''
            for k in range(len(x_y_weight_list)):
                if(k in sub_indices):
                    fig.data[k].line = {'color': colory, 'width': list(g.edges.data('weight'))[k][2]}
                    #trace.push(fig.data[k])
                    fig.data.insert(-2, fig.data.pop(i))
                else:
                    fig.data[k].line = {'color': 'black', 'width': list(g.edges.data('weight'))[k][2]}
            '''
    def prev_btn_event_handler(self,btn_object):
        #scatter.marker.color = list_of_pure_color_lists[t]
        self.time_slider.value -= 1

    def next_btn_event_handler(self,btn_object):
        #scatter.marker.color = list_of_pure_color_lists[t]  
        self.time_slider.value += 1

    def on_value_change(self,t):
        with self.fig.batch_update():
            self.scatter.marker.color = self.list_of_pure_color_lists[t['owner'].value]    

            


#display(HBox([prev_btn,time_slider, next_btn]))            
            
            
def get_fig_data_from_own_G(own_G_1):
    g = own_G_1.G

    # Get a layout for the nodes according to some algorithm.
    # See https://networkx.github.io/documentation/stable/reference/drawing.html#layout
    # for alternative algorithms that are available.
    # Set random_state (default=None) if you want the layout to be deterministic
    # and repeatable.
    node_positions = own_G_1.pos #nx.spring_layout(g)
    node_txt = own_G_1.txt


    # The nodes will be plotted as a scatter plot of markers with their names
    # above each circle:

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='top center',
        textfont=dict(
            family='arial',
            size=node_description_size,
            color='rgb(0,0,0)'
        ),
        hoverinfo='none',
        marker=go.scatter.Marker(
                showscale=False,
                color=init_node_color,
                size=init_node_size,
                line=dict(width=2, color='black')) 
                 ) #,line=go.Line(width=1, color='rgb(0,0,0)'))


    for node in node_positions:
        #print("node_positions[node]: ", node_positions[node])
        x, y = node_positions[node]
        #y*= 2
        txt_pos = own_G_1.txt_pos[node]

        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        #node_trace['textposition'] += txt_pos
        
        if(node in node_txt.keys()):
            txt = node_txt[node]
            node_trace['text'] += tuple([txt])
        else:
            node_trace['text'] += tuple([""])

    traces={}
    i=0
    for edge in g.edges.data('weight'):
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]

        traces['trace_' + str(i)]=go.Scatter(      
            x= tuple([x0, x1, None]),
            y= tuple([y0, y1, None]),
            line=dict(     
                           color=init_weight_color,
                           width=abs(edge[2]) ))
        i+=1
    data=list(traces.values())    
    #data.append(button_trace)
    data.append(node_trace)
    
    return data


def get_color_list(graph_list, time_step):
    own_G_1 = graph_list[time_step]
    color_list =  []
    for key, value in own_G_1.activations.items():
        temp = [key,value]
        color = bipolar_number_to_color_255(temp[1])
        color = 'rgba' + str(color)
        color_list.append(color)
    return [{'marker.color': [color_list]}]

def get_pure_color_list(graph_list, time_step):
    own_G_1 = graph_list[time_step]
    color_list =  []
    for key, value in own_G_1.activations.items():
        temp = [key,value]
        color = bipolar_number_to_color_255(temp[1])
        color = 'rgba' + str(color)
        color_list.append(color)
    return color_list

def get_list_of_pure_color_lists(graph_list): 
    list_of_pure_color_lists = []
    n_time_steps = len(graph_list)
    for i in range(n_time_steps):
        color_list_i = get_pure_color_list(graph_list, time_step=i)
        list_of_pure_color_lists.append(color_list_i)
    return list_of_pure_color_lists





################
### 1) Initial: Data + Figure
#####################

def plot_network_from_graph_list(graph_list):
        own_G_1 = graph_list[0]
        data = get_fig_data_from_own_G(own_G_1)
        # ... and put into Figure
        fig=go.FigureWidget(data)
        fig.layout.height=1000
        orig_fig_data = copy(list(fig.data)[:] )
        n_time_steps = len(graph_list)



        ################
        ### 2) Interaction
        #####################
        # Now prepare data you want to interact with:
        scatter = fig.data[-1]
        color_list = get_pure_color_list(graph_list, 0)
        scatter.marker.color = color_list
        node_positions = own_G_1.pos
        scatter.marker.size = [init_node_size] * len(node_positions)
        scatter.textposition= list(own_G_1.txt_pos.values())
        #fig.layout.hovermode = 'closest'
        #fig.layout.height=400
        #fig.layout.width=2000

        layer_descriptions = own_G_1.layer_descriptions

        for layer_i in layer_descriptions.items():    
            ## Test layer description
            x, y = layer_i[1][1]
            txt = layer_i[1][0]
            txt_pos = 'top center'

            scatter['x'] += tuple([x])
            scatter['y'] += tuple([y])
            scatter['text'] += tuple([txt])

        layer_operations = own_G_1.layer_operation
        for layer_i in layer_operations.items():    
            ## Test layer description
            x, y = layer_i[1][1]
            txt = layer_i[1][0]
            #txt_pos = 'middle center'
            scatter['x'] += tuple([x])
            scatter['y'] += tuple([y])
            scatter['text'] += tuple([txt])


        # Interaction within one time-step: change node-size and weight-emphasis
        plt_int = plotly_interaction(graph_list, fig, scatter)
        scatter.on_click(plt_int.update_point)

        fig.update_layout(showlegend=False)
        #fig.update_xaxes(range=[5,85], autorange=False)
        #fig.update_yaxes(range=[-0.05,0.5], autorange=False)

        display(plt_int.time_control_fig)
        display(fig)