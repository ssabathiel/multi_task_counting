# Results for ALIFE paper



#### 1. Representation for Entity-to-number-word mapping ('1-1 correspondence'):

Identified nodes in the visual representation, which are turned on when there is no entity to count and turned off whenever there is an entity to count.
These nodes have inhibiting connections to the output of the number words.

See example on 'Count all objects' task:

![entity_repr_count_all_objects](.\entity_repr_count_all_objects.png)



##### Is this representation 'abstract' in the sense that it is independent of entity and number word?

###### Is this representation independent of the number word? 

Yes: 
1) there are inhibiting connections (black dots in the weight matrix) from the same 'entity-node' to all the number words 
2) see tuning curve below



###### Is this representation independent of the entity?

To answer this question we look at several instances of entity/no-entity and different tasks.

...

Even more instances of entity/no-entity and different tasks, to identify nodes more reliably.

Inline-style: 
![alt text](C:\Users\silvests\Embodied_counting\Results\ALIFE_paper\entity_repr_many_examples.png)











Check if identified node represents entities **independent of the entity/task and number word** **NUMERICALLY**: Tuning Curves

![entity_repr_num](C:\Users\silvests\Embodied_counting\Results\ALIFE_paper\entity_repr_num.png)



![entity_repr_num_2](C:\Users\silvests\Embodied_counting\Results\ALIFE_paper\entity_repr_num_2.png)



```

```













#### 2. Representation for the number sequence ('Stable-order-principle')

Even though the representations are not localized/are distributed they are shared among all tasks. see fig below.

###### Number representation independent of entity.

![number_repr_size_encoded_give_n](C:\Users\silvests\Embodied_counting\Results\ALIFE_paper\number_repr_size_encoded_give_n.PNG)

