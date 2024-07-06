# Created by Eosandra Grund
# contains analysis functions to analyse results saved as interactions
# average message length
# 


# ZLA significance score as described by Rita et al. (2020)
# ********************

import torch
from language_analysis_local import MessageLengthHierarchical
from utils.analysis_from_interaction import *

def mean_message_length_from_interaction(interaction):
    """ Calculates the average message length, with only accounting for each individual message one time. 
    
    :param interaction: interaction (EGG class)
    """
    messages = interaction.message.argmax(dim=-1)
    unique_messages = torch.unique(messages,dim=0)
    message_length = MessageLengthHierarchical.compute_message_length(unique_messages)
    av_message_length = torch.mean(message_length.float())
    return av_message_length

def mean_weighted_message_length_from_interaction(interaction):
    """ average length of the messages weighted by their generation frequency (how often does the agent generate this message)
    
    :param interaction: interaction (EGG class)
    """
    messages = interaction.message.argmax(dim=-1)
    message_length = MessageLengthHierarchical.compute_message_length(messages)
    av_message_length = torch.mean(message_length.float())
    return av_message_length

def mean_weighted_message_length(length,frequency): # TODO analyse context x concept counts and if it matches this calculation but should
    """ calculates the average length of the messages wieghted by their generation frequency from length and frequency. 
    
    :param length: torch.Tensor with shape = (n,) contains the length of each message
    :param frequency: torch.Tensor with shape = (n,) contains the frequency of each message
    """
    return torch.sum(((length * frequency)/torch.sum(frequency)).float())


def ZLA_significance_score(interaction,num_permutations = 1000):
    """ Calculates to which degree mean_weighted_message_length is lower than a random permutation of its frequency length mapping 
    
    :param interaction: interaction (EGG class)
    :param num_permutations: int
    """

    # get message frequencies and message_length
    messages = interaction.message.argmax(dim=-1)
    unique_messages, frequencies = torch.unique(messages,dim=0,return_counts=True)
    message_length = MessageLengthHierarchical.compute_message_length(unique_messages)

    L_type = torch.mean(message_length.float()) # mean message length
    original_L_token = mean_weighted_message_length_from_interaction(interaction)

    # random permutations of frequency-length mapping, then calulate their L_token
    permuted_L_tokens = []
    for _ in range(num_permutations):
        permuted_indices = torch.randperm(message_length.shape[0])
        permuted_lengths = message_length[permuted_indices]
        permuted_L_token = mean_weighted_message_length(permuted_lengths,frequencies)
        permuted_L_tokens.append(permuted_L_token)

    # calculate p_value
    pZLA = (torch.tensor(permuted_L_tokens) <= original_L_token).float().mean()

    score_dict = {'mean_message_length':L_type,
                  'mean_weighted_message_length':original_L_token,
                  'Pzla':pZLA}
    return score_dict

# positional encoding as described by Rita et al. (2020)
# *************************************

def symbol_informativeness(listener,interaction,eos_token = 0.0):
    """ Calculates for each symbol if it is informative or not by switching its value with a random other symbols value for the same position (except eos) and observing if the prediction of the listener changes. 
    
    :param listener:
    :param interaction: interaction (EGG class)
    :param eos_token: float

    Example switch for position 1 in message with message length 2, switch max = 2 with switch_index = 1 (here only option as eos an max are excluded)
    original_message = tensor([[0.2,0.3,0.5],[0.3,0.5,0.2]]) -> message = [2,1]
    manipulated_message = tensor([[0.2,0.5,0.3],[0.3,0.5,0.2]]) -> message = [1,1]
    """

    messages = interaction.message
    num_messages, message_length, vocab_size = messages.shape

    # values to be switched
    max_values, original_messages = messages.max(dim=-1)

    eos_mask_total = eos_token != original_messages
    eos_cumulative = []
    Lambda_m_k_list = []

    for i in range(message_length-1): # index = -1 is always eos and not changed. 

        # only take ones that did not have eos yet
        eos_mask = eos_mask_total[:,:i+1].sum(dim=1).unsqueeze(1) == i+1
        eos_cumulative.append(eos_mask)

        # pick random new symbol index to switch with
        switch_index = torch.randint(1,vocab_size,(num_messages,))

        # make sure switch_index != original_messages
        same_indices = (switch_index == original_messages[:,i])
        while same_indices.any():
            switch_index[same_indices] = torch.randint(1, vocab_size, (same_indices.sum().item(),))
            same_indices = (switch_index == original_messages[:,i])

        # values to switch with
        new_value = torch.gather(messages[:,0,:],-1,switch_index.unsqueeze(-1)).squeeze()

        messages_manipulated = messages.clone()

        # if eosed, take old value in so no switch
        new_value_without_eos = torch.where(eos_mask.squeeze(), new_value,max_values[:,i])
        max_value_without_eos = torch.where(eos_mask.squeeze(),max_values[:,i],new_value)

        # switch value i of all messages to new symbol
        messages_manipulated[:,0].scatter_(-1,original_messages[:,i].unsqueeze(-1),new_value_without_eos.unsqueeze(-1))
        messages_manipulated[:,0].scatter_(-1,switch_index.unsqueeze(-1),max_value_without_eos.unsqueeze(-1))

        prediction_manipulated = (listener(messages_manipulated,interaction.receiver_input) > 0).float() # bigger 0 means thinks its a concept object
        prediction = (interaction.receiver_output > 0).float()

        # compare whether same classification not same values
        Lambda_m_k_position = (prediction != prediction_manipulated).sum(dim=(1,2)).unsqueeze(1).bool() # (num_messages, 1 )
        Lambda_m_k_list.append(Lambda_m_k_position)

    Lambda_m_k = torch.cat(Lambda_m_k_list,dim=1)
    eos_cumulative_tensor = torch.cat(eos_cumulative,dim=1)

    return Lambda_m_k, eos_cumulative_tensor # both should be the same shape

def positional_encoding(Lambda_m_k,eos_cumulative):
    """ Calculates a vector Lambda_dot_k, that contains the proportions of informative symbols for each position in messages. Lambda_m_k. Excludes eos 
    
    :param Lambda_m_k: torch.Tensor with shape (num_messages,message_lenght - 1)
    :param eos_cumulative: torch.Tensor with shape (num_messages,message_lenght - 1)
    """
    nr_non_eos_per_position = eos_cumulative.sum(dim=0)
    nr_informative_per_position = Lambda_m_k.sum(dim=0)
    return nr_informative_per_position / nr_non_eos_per_position

def effective_length(Lambda_m_k,eos_cumulative):
    """ Measures the mean number of informative symbols by message.
    
    :param Lambda_m_k: torch.Tensor with shape (num_messages,message_lenght - 1)
    :param eos_cumulative: torch.Tensor with shape (num_messages,message_lenght - 1)
    """
    nr_non_eos_per_message = eos_cumulative.sum(dim=1)
    nr_informative_per_message = Lambda_m_k.sum(dim=1)
    return nr_informative_per_message / nr_non_eos_per_message
    
def information_density(Lambda_m_k,eos_cumulative):
    """ Measures the fraction of informative symbols in a language. 
    
    :param Lambda_m_k: torch.Tensor with shape (num_messages,message_lenght - 1)
    :param eos_cumulative: torch.Tensor with shape (num_messages,message_lenght - 1)
    """
    nr_non_eos = eos_cumulative.sum()
    nr_informative = Lambda_m_k.sum()
    return nr_informative / nr_non_eos

def information_analysis(listener,interaction,eos_token = 0.0):
    """ Calculates positional_encoding, effective_length and information_density. 
    
    :param listener:
    :param interaction: interaction
    :param eos_token: float
    """

    Lambda_m_k, eos_mask = symbol_informativeness(listener,interaction,eos_token)

    Lambda_dot_k = positional_encoding(Lambda_m_k, eos_mask)
    L_eff = effective_length(Lambda_m_k, eos_mask)
    rho_inf = information_density(Lambda_m_k,eos_mask)

    return_dict = {
        "positional_encoding" : Lambda_dot_k,
        "effective_length" : L_eff,
        "information_density" : rho_inf
    }
    return return_dict


# other functions I might need
#**************************

def load_interaction(path,setting,nr=0,n_epochs=300):
    """ loads an interaction given the path, setting, nr and how many epochs where done (Path before setting)
    
    :param path: str
    :param setting: str
    :param nr: int
    :param n_epochs: int
    """
    # select first run
    path_to_run = path + '/' + str(setting) +'/' + str(nr) + '/'
    path_to_interaction_train = (path_to_run + 'interactions/train/epoch_' + str(n_epochs) + '/interaction_gpu0')
    path_to_interaction_val = (path_to_run + 'interactions/validation/epoch_' + str(n_epochs) + '/interaction_gpu0')
    interaction = torch.load(path_to_interaction_train)
    print(path_to_interaction_train)
    return interaction

def retrieve_messages(interaction,as_list = True):
    """ retrieves the messages from an interaction 
    
    :param interaction: interaction (EGG class)
    :param as_list: bool
    """
    messages = interaction.message.argmax(dim=-1)
    if not as_list:
        return messages
    return [msg.tolist() for msg in messages]

def retrieve_concepts_context(interaction,n_values):
    """ retrieves Concepts and context conditions from an interaction
    
    :param interaction: interaction (EGG class)
    :param n_values: int
    """
    sender_input = interaction.sender_input
    print(sender_input.shape)
    n_targets = int(sender_input.shape[1]/2)
    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects, all_targets=True)
    concepts = list(zip(objects, fixed))

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)
    
    return concepts, context_conds