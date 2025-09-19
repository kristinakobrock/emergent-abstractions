# copied and adapted from https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/utils/analysis_from_interaction.py

from egg.core.language_analysis import calc_entropy, _hashable_tensor, Disent
from sklearn.metrics import normalized_mutual_info_score
from language_analysis_local import MessageLengthHierarchical
from collections import Counter
import numpy as np
import itertools
import random
import torch
import torch.nn.functional as F


def k_hot_to_attributes(khots, dimsize):
    """
    Decodes many-hot-represented objects to an easy-to-interpret version.
    E.g. [0, 0, 1, 1, 0, 0, 1, 0, 0] -> [2, 0, 0]
    """
    base_count = 0
    n_attributes = khots.shape[-1] // dimsize
    attributes = np.zeros((len(khots), len(khots[0]), n_attributes))
    for att in range(n_attributes):
        attributes[:, :, att] = np.argmax(khots[:, :, base_count:base_count + dimsize], axis=2)
        base_count = base_count + dimsize
    return attributes


def retrieve_fixed_vectors(target_objects):
    """
    Reconstructs fixed vectors given a list of (decoded) target objects.
    """
    n_attributes = target_objects.shape[2]
    # compare target objects to each other to find out whether and in which attribute they differ
    fixed_vectors = []
    for target_objs in target_objects:
        fixed = np.ones(n_attributes)  # (1, 1, 1) -> all fixed
        for idx, target_object in enumerate(target_objs):
            # take first target_object as baseline for comparison (NOTE: could also be random)
            if idx == 0:
                concept = target_object
            else:
                for i, attribute in enumerate(target_object):
                    # if values mismatch, then the attribute is not fixed, i.e. 0 in fixed vector
                    if attribute != concept[i]:
                        fixed[i] = 0
        fixed_vectors.append(fixed)
    return fixed_vectors


def convert_fixed_to_intentions(fixed_vectors):
    """
    NOTE: not needed right now
    fixed vectors are 0: irrelevant, 1: relevant
    intentions are 1: irrelevant, 0: relevant
    """
    intentions = []
    for fixed in fixed_vectors:
        intention = np.zeros(len(fixed))
        for i, att in enumerate(fixed):
            if att == 0:
                intention[i] = 1
        intentions.append(intention)
    return np.asarray(intentions)


def retrieve_concepts_sampling(target_objects, all_targets=False):
    """
    Builds concept representations consisting of one sampled target object and a fixed vector.
    """
    fixed_vectors = retrieve_fixed_vectors(target_objects)
    if all_targets:
        target_objects_sampled = target_objects
    else:
        target_objects_sampled = [random.choice(target_object) for target_object in target_objects]
    return (np.asarray(target_objects_sampled), np.asarray(fixed_vectors))


def joint_entropy(xs, ys):
    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    return calc_entropy(xys)


def retrieve_context_condition(targets, fixed, distractors):
    """returns the context condition given a list of targets and a list of distractors (from interaction)"""
    context_conds = []
    # go through all observations
    for i, t_obj in enumerate(targets):
        # consider first target and first distractor
        shared = 0
        # go through attributes
        if t_obj.ndim > 1:
            for k, attr in enumerate(t_obj[0]):
                # if target attribute was fixed:
                if fixed[i][k] == 1:
                    # shared = np.zeros(len(t_obj))
                    # go through distractors
                    # for dist_obj in distractors[i]:
                    # compare target attribute with distractor attribute
                    if attr == distractors[i][0][k]:
                        # count shared attributes
                        shared = shared + 1
        else:
            for k, attr in enumerate(t_obj):
                if fixed[i][k] == 1:
                    if attr == distractors[i][0][k]:
                        shared = shared + 1
        # print("target", t_obj, "fixed", fixed[i], "distractors", distractors[i][0], "shared", shared)
        context_conds.append(shared)
    return context_conds


def trim_tensor(tensor):
    # Find the index of the first zero
    zero_indices = (tensor == 0).nonzero(as_tuple=True)[0]
    # Trim the tensor up to the index before the zero
    if len(zero_indices) > 0:
        # Take the first zero index and trim the tensor
        first_zero_index = zero_indices[0].item()
        trimmed_tensor = tensor[:first_zero_index]
    else:
        # If there is no zero, return the full tensor
        trimmed_tensor = tensor
    return trimmed_tensor


def bosdis(interaction, n_dims, n_values, vocab_size):
    """
    calculate bag-of-symbol disentanglement for all concept and context conditions
    """
    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)
    # add one such that zero becomes an empty attribute for the calculation (_)
    objects = objects + 1
    concepts = torch.from_numpy(objects * (np.array(fixed)))

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    distractor_objects = distractor_objects + 1
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)

    # get messages from interaction
    messages = interaction.message.argmax(dim=-1)

    # sum of fixed vectors gives the specificity of the concept (all attributes fixed means
    # specific concept, one attribute fixed means generic concept)
    # n_relevant_idx stores the indices of the concepts on a specific level of abstraction
    n_relevant_idx = [np.where(np.sum(np.array(fixed), axis=1) == i)[0] for i in range(1, n_dims + 1)]

    context_cond_idx = [np.where(np.array(context_conds) == i)[0] for i in range(0, n_dims)]

    # Concept-context dependent Entropies:
    # go through concept conditions
    conceptxcontext_idx = []
    for i in range(len(n_relevant_idx)):
        # go through context conditions
        for j in range(len(context_cond_idx)):
            # only keep shared entries for each concept-context condition
            shared_elements = [elem for elem in n_relevant_idx[i] if elem in context_cond_idx[j]]
            conceptxcontext_idx.append(shared_elements)

    # Bosdis for each concept and context condition
    bosdis_concept_x_context = np.array(
        [Disent.bosdis(concepts[concept_x_context], messages[concept_x_context], vocab_size) for concept_x_context in
         conceptxcontext_idx])

    return bosdis_concept_x_context


def posdis(interaction, n_dims, n_values, vocab_size):
    """
    calculate positional disentanglement for all concept and context conditions
    """
    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)
    # add one such that zero becomes an empty attribute for the calculation (_)
    objects = objects + 1
    concepts = torch.from_numpy(objects * (np.array(fixed)))

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    distractor_objects = distractor_objects + 1
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)

    # get messages from interaction
    messages = interaction.message.argmax(dim=-1)

    # sum of fixed vectors gives the specificity of the concept (all attributes fixed means
    # specific concept, one attribute fixed means generic concept)
    # n_relevant_idx stores the indices of the concepts on a specific level of abstraction
    n_relevant_idx = [np.where(np.sum(np.array(fixed), axis=1) == i)[0] for i in range(1, n_dims + 1)]

    context_cond_idx = [np.where(np.array(context_conds) == i)[0] for i in range(0, n_dims)]

    # Concept-context dependent Entropies:
    # go through concept conditions
    conceptxcontext_idx = []
    for i in range(len(n_relevant_idx)):
        # go through context conditions
        for j in range(len(context_cond_idx)):
            # only keep shared entries for each concept-context condition
            shared_elements = [elem for elem in n_relevant_idx[i] if elem in context_cond_idx[j]]
            conceptxcontext_idx.append(shared_elements)

    # Posdis for each concept and context condition
    posdis_concept_x_context = np.array(
        [Disent.posdis(concepts[concept_x_context], messages[concept_x_context]) for concept_x_context in
         conceptxcontext_idx])

    return posdis_concept_x_context


def information_scores(interaction, n_dims, n_values, normalizer="arithmetic", is_gumbel=True, trim_eos=False,
                       max_mess_len=None):
    """calculate entropy scores: mutual information (MI), effectiveness and consistency. 
    
    :param interaction: interaction (EGG class)
    :param n_dims: number of input dimensions, e.g. D(3,4) --> 3 dimensions
    :param n_values: size of each dimension, e.g. D(3,4) --> 4 values
    :param normalizer: normalizer can be either "arithmetic" -H(M) + H(C)- or "joint" -H(M,C)-
    :return: NMI, NMI per level, effectiveness, effectiveness per level, consistency, consistency per level
    """

    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)
    # add one such that zero becomes an empty attribute for the calculation (_)
    objects = objects + 1
    concepts = torch.from_numpy(objects * (np.array(fixed)))
    #concepts = sender_input.reshape(-1, np.prod(sender_input.shape[1:]))
    #print(type(concepts), concepts.shape)

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    distractor_objects = distractor_objects + 1
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)

    # get messages from interaction
    messages = interaction.message.argmax(dim=-1) if is_gumbel is True else interaction.message.argmax(dim=2)
    if trim_eos:
        messages = [trim_tensor(message) for message in messages]
        # Pad tensors to make them uniform length
        padded_tensors = [F.pad(t, pad=(0, max_mess_len - t.size(0))) for t in messages]
        # Stack into a single tensor
        messages = torch.stack(padded_tensors)
    # print(type(messages[0]), messages.shape)

    # Entropies:
    # H(m), H(c), H(m,c)
    m_entropy = calc_entropy(messages)
    c_entropy = calc_entropy(concepts)
    joint_mc_entropy = joint_entropy(messages, concepts)

    # Concept-dependent Entropies:
    # sum of fixed vectors gives the specificity of the concept (all attributes fixed means
    # specific concept, one attribute fixed means generic concept)
    # n_relevant_idx stores the indices of the concepts on a specific level of abstraction
    n_relevant_idx = [np.where(np.sum(np.array(fixed), axis=1) == i)[0] for i in range(1, n_dims + 1)]
    # print("first message", messages[n_relevant_idx[0]])
    # print("ent first level", calc_entropy(messages[n_relevant_idx[0]]))
    # print("second level message", messages[n_relevant_idx[1]])
    # print("ent second level", calc_entropy(messages[n_relevant_idx[1]]))
    # for n_relevant in n_relevant_idx:
    #     print(calc_entropy(messages[n_relevant]))
    #
    # print("first concept", concepts[n_relevant_idx[0]][0])
    # print("ent first level", calc_entropy(concepts[n_relevant_idx[0]]))
    # print("ent second level", calc_entropy(concepts[n_relevant_idx[1]]))

    # H(m), H(c), H(m,c) for each level of abstraction
    m_entropy_hierarchical = np.array([calc_entropy(messages[n_relevant]) for n_relevant in n_relevant_idx])
    # print("m_entr_hier", m_entropy_hierarchical)
    c_entropy_hierarchical = np.array([calc_entropy(concepts[n_relevant]) for n_relevant in n_relevant_idx])
    # print("c_entr_hier", c_entropy_hierarchical)
    joint_entropy_hierarchical = np.array([joint_entropy(messages[n_relevant], concepts[n_relevant])
                                           for n_relevant in n_relevant_idx])
    # print("j_entr_hier", joint_entropy_hierarchical)

    # Context-dependent Entropies:
    context_cond_idx = [np.where(np.array(context_conds) == i)[0] for i in range(0, n_dims)]
    # H(m), H(c), H(m,c) for each context condition
    m_entropy_context_dep = np.array([calc_entropy(messages[context_cond]) for context_cond in context_cond_idx])
    c_entropy_context_dep = np.array([calc_entropy(concepts[context_cond]) for context_cond in context_cond_idx])
    joint_entropy_context_dep = np.array([joint_entropy(messages[context_cond], concepts[context_cond])
                                          for context_cond in context_cond_idx])

    # Concept-context dependent Entropies:
    # go through concept conditions
    conceptxcontext_idx = []
    for i in range(len(n_relevant_idx)):
        # go through context conditions
        for j in range(len(context_cond_idx)):
            # only keep shared entries for each concept-context condition
            shared_elements = [elem for elem in n_relevant_idx[i] if elem in context_cond_idx[j]]
            conceptxcontext_idx.append(shared_elements)

    # H(m), H(c), H(m,c) for each concept and context condition
    m_entropy_concept_x_context = np.array(
        [calc_entropy(messages[concept_x_context]) for concept_x_context in conceptxcontext_idx])
    c_entropy_concept_x_context = np.array(
        [calc_entropy(concepts[concept_x_context]) for concept_x_context in conceptxcontext_idx])
    joint_entropy_concept_x_context = np.array([joint_entropy(messages[concept_x_context], concepts[concept_x_context])
                                                for concept_x_context in conceptxcontext_idx])

    # Normalized scores: NMI, consistency, effectiveness
    if normalizer == "arithmetic":
        normalizer = 0.5 * (m_entropy + c_entropy)
        normalizer_hierarchical = 0.5 * (m_entropy_hierarchical + c_entropy_hierarchical)
        normalizer_context_dep = 0.5 * (m_entropy_context_dep + c_entropy_context_dep)
        normalizer_conc_x_cont = 0.5 * (m_entropy_concept_x_context + c_entropy_concept_x_context)
    elif normalizer == "joint":
        normalizer = joint_mc_entropy
        normalizer_hierarchical = joint_entropy_hierarchical
        normalizer_context_dep = joint_entropy_context_dep
        normalizer_conc_x_cont = joint_entropy_concept_x_context
    else:
        raise AttributeError("Unknown normalizer")

    # normalized mutual information: H(m) - H(m|c) / normalizer, H(m|c)=H(m,c)-H(c)
    normalized_MI = (m_entropy + c_entropy - joint_mc_entropy) / normalizer
    # print("normalizer", normalizer_hierarchical)
    # print("entropies", (m_entropy_hierarchical + c_entropy_hierarchical - joint_entropy_hierarchical))
    # print("entropies 2", (m_entropy_hierarchical - (c_entropy_hierarchical - joint_entropy_hierarchical)))
    normalized_MI_hierarchical = ((m_entropy_hierarchical + c_entropy_hierarchical - joint_entropy_hierarchical)
                                  / normalizer_hierarchical)
    # print("MI hier", normalized_MI_hierarchical)
    normalized_MI_context_dep = ((m_entropy_context_dep + c_entropy_context_dep - joint_entropy_context_dep)
                                 / normalizer_context_dep)
    normalized_MI_conc_x_cont = (
            (m_entropy_concept_x_context + c_entropy_concept_x_context - joint_entropy_concept_x_context)
            / normalizer_conc_x_cont)

    # normalized version of h(c|m), i.e. h(c|m)/h(c)
    normalized_effectiveness = (joint_mc_entropy - m_entropy) / c_entropy
    normalized_effectiveness_hierarchical = ((joint_entropy_hierarchical - m_entropy_hierarchical)
                                             / c_entropy_hierarchical)
    normalized_effectiveness_context_dep = ((joint_entropy_context_dep - m_entropy_context_dep)
                                            / c_entropy_context_dep)
    normalized_effectiveness_conc_x_cont = ((joint_entropy_concept_x_context - m_entropy_concept_x_context)
                                            / c_entropy_concept_x_context)

    # normalized version of h(m|c), i.e. h(m|c)/h(m)
    normalized_consistency = (joint_mc_entropy - c_entropy) / m_entropy
    normalized_consistency_hierarchical = (joint_entropy_hierarchical - c_entropy_hierarchical) / m_entropy_hierarchical
    # print("cons hier", normalized_consistency_hierarchical)
    normalized_consistency_context_dep = (joint_entropy_context_dep - c_entropy_context_dep) / m_entropy_context_dep
    normalized_consistency_conc_x_cont = (joint_entropy_concept_x_context - c_entropy_concept_x_context) / m_entropy_concept_x_context

    score_dict = {'normalized_mutual_info': normalized_MI,
                  'normalized_mutual_info_hierarchical': normalized_MI_hierarchical,
                  'normalized_mutual_info_context_dep': normalized_MI_context_dep,
                  'normalized_mutual_info_concept_x_context': normalized_MI_conc_x_cont,
                  'effectiveness': 1 - normalized_effectiveness,
                  'effectiveness_hierarchical': 1 - normalized_effectiveness_hierarchical,
                  'effectiveness_context_dep': 1 - normalized_effectiveness_context_dep,
                  'effectiveness_concept_x_context': 1 - normalized_effectiveness_conc_x_cont,
                  'consistency': 1 - normalized_consistency,
                  'consistency_hierarchical': 1 - normalized_consistency_hierarchical,
                  'consistency_context_dep': 1 - normalized_consistency_context_dep,
                  'consistency_concept_x_context': 1 - normalized_consistency_conc_x_cont
                  }
    return score_dict


def information_scores_new(interaction, n_dims, n_values, normalizer="arithmetic", is_gumbel=True, trim_eos=False,
                       max_mess_len=None):
    """calculate entropy scores: mutual information (MI), effectiveness and consistency.

    :param interaction: interaction (EGG class)
    :param n_dims: number of input dimensions, e.g. D(3,4) --> 3 dimensions
    :param n_values: size of each dimension, e.g. D(3,4) --> 4 values
    :param normalizer: normalizer can be either "arithmetic" -H(M) + H(C)- or "joint" -H(M,C)-
    :return: NMI, NMI per level, effectiveness, effectiveness per level, consistency, consistency per level
    """

    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)
    # add one such that zero becomes an empty attribute for the calculation (_)
    objects = objects + 1
    concepts = torch.from_numpy(objects * (np.array(fixed)))

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    distractor_objects = distractor_objects + 1
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)

    # get messages from interaction
    messages = interaction.message.argmax(dim=-1) if is_gumbel is True else interaction.message.argmax(dim=2)
    if trim_eos:
        messages = [trim_tensor(message) for message in messages]
        # Pad tensors to make them uniform length
        padded_tensors = [F.pad(t, pad=(0, max_mess_len - t.size(0))) for t in messages]
        # Stack into a single tensor
        messages = torch.stack(padded_tensors)

    # Entropies:
    # H(m), H(c), H(m,c)
    m_entropy = calc_entropy(messages)
    c_entropy = calc_entropy(concepts)
    joint_mc_entropy = joint_entropy(messages, concepts)

    # Concept-dependent Entropies:
    # sum of fixed vectors gives the specificity of the concept (all attributes fixed means
    # specific concept, one attribute fixed means generic concept)
    # n_relevant_idx stores the indices of the concepts on a specific level of abstraction
    n_relevant_idx = [np.where(np.sum(np.array(fixed), axis=1) == i)[0] for i in range(1, n_dims + 1)]
    # print("first message", messages[n_relevant_idx[0]])
    # print("ent first level", calc_entropy(messages[n_relevant_idx[0]]))
    # print("second level message", messages[n_relevant_idx[1]])
    # print("ent second level", calc_entropy(messages[n_relevant_idx[1]]))
    # for n_relevant in n_relevant_idx:
    #     print(calc_entropy(messages[n_relevant]))
    #
    # print("first concept", concepts[n_relevant_idx[0]])
    # print("ent first level", calc_entropy(concepts[n_relevant_idx[0]]))
    # print("ent second level", calc_entropy(concepts[n_relevant_idx[1]]))

    # H(m), H(c), H(m,c) for each level of abstraction
    m_entropy_hierarchical = np.array([calc_entropy(messages[n_relevant]) for n_relevant in n_relevant_idx])
    # print("m_entr_hier", m_entropy_hierarchical)
    c_entropy_hierarchical = np.array([calc_entropy(concepts[n_relevant]) for n_relevant in n_relevant_idx])
    # print("c_entr_hier", c_entropy_hierarchical)
    joint_entropy_hierarchical = np.array([joint_entropy(messages[n_relevant], concepts[n_relevant])
                                           for n_relevant in n_relevant_idx])
    # print("j_entr_hier", joint_entropy_hierarchical)

    # Context-dependent Entropies:
    context_cond_idx = [np.where(np.array(context_conds) == i)[0] for i in range(0, n_dims)]
    # H(m), H(c), H(m,c) for each context condition
    m_entropy_context_dep = np.array([calc_entropy(messages[context_cond]) for context_cond in context_cond_idx])
    c_entropy_context_dep = np.array([calc_entropy(concepts[context_cond]) for context_cond in context_cond_idx])
    joint_entropy_context_dep = np.array([joint_entropy(messages[context_cond], concepts[context_cond])
                                          for context_cond in context_cond_idx])

    # Concept-context dependent Entropies:
    # go through concept conditions
    conceptxcontext_idx = []
    for i in range(len(n_relevant_idx)):
        # go through context conditions
        for j in range(len(context_cond_idx)):
            # only keep shared entries for each concept-context condition
            shared_elements = [elem for elem in n_relevant_idx[i] if elem in context_cond_idx[j]]
            conceptxcontext_idx.append(shared_elements)

    # H(m), H(c), H(m,c) for each concept and context condition
    m_entropy_concept_x_context = np.array(
        [calc_entropy(messages[concept_x_context]) for concept_x_context in conceptxcontext_idx])
    c_entropy_concept_x_context = np.array(
        [calc_entropy(concepts[concept_x_context]) for concept_x_context in conceptxcontext_idx])
    joint_entropy_concept_x_context = np.array([joint_entropy(messages[concept_x_context], concepts[concept_x_context])
                                                for concept_x_context in conceptxcontext_idx])

    # Normalized scores: NMI, consistency, effectiveness
    if normalizer == "arithmetic":
        normalizer = 0.5 * (m_entropy + c_entropy)
        normalizer_hierarchical = 0.5 * (m_entropy_hierarchical + c_entropy_hierarchical)
        normalizer_context_dep = 0.5 * (m_entropy_context_dep + c_entropy_context_dep)
        normalizer_conc_x_cont = 0.5 * (m_entropy_concept_x_context + c_entropy_concept_x_context)
    elif normalizer == "joint":
        normalizer = joint_mc_entropy
        normalizer_hierarchical = joint_entropy_hierarchical
        normalizer_context_dep = joint_entropy_context_dep
        normalizer_conc_x_cont = joint_entropy_concept_x_context
    else:
        raise AttributeError("Unknown normalizer")

    # normalized mutual information: H(m) - H(m|c) / normalizer, H(m|c)=H(m,c)-H(c)
    normalized_MI = (m_entropy - (c_entropy - joint_mc_entropy)) / normalizer
    # print("NMI 1", (m_entropy + c_entropy - joint_mc_entropy) / normalizer)
    # print("NMI 2", (m_entropy - (c_entropy - joint_mc_entropy)) / normalizer)
    # print("normalizer", normalizer_hierarchical)
    # print("entropies", (m_entropy_hierarchical + c_entropy_hierarchical - joint_entropy_hierarchical))
    # print("entropies 2", (m_entropy_hierarchical - (c_entropy_hierarchical - joint_entropy_hierarchical)))
    normalized_MI_hierarchical = ((m_entropy_hierarchical - (c_entropy_hierarchical - joint_entropy_hierarchical))
                                  / normalizer_hierarchical)
    # print("MI hier", normalized_MI_hierarchical)
    normalized_MI_context_dep = ((m_entropy_context_dep - (c_entropy_context_dep - joint_entropy_context_dep))
                                 / normalizer_context_dep)
    normalized_MI_conc_x_cont = (
            (m_entropy_concept_x_context - (c_entropy_concept_x_context - joint_entropy_concept_x_context))
            / normalizer_conc_x_cont)

    # normalized version of h(c|m), i.e. h(c|m)/h(c)
    normalized_effectiveness = (joint_mc_entropy - m_entropy) / c_entropy
    normalized_effectiveness_hierarchical = ((joint_entropy_hierarchical - m_entropy_hierarchical)
                                             / c_entropy_hierarchical)
    normalized_effectiveness_context_dep = ((joint_entropy_context_dep - m_entropy_context_dep)
                                            / c_entropy_context_dep)
    normalized_effectiveness_conc_x_cont = ((joint_entropy_concept_x_context - m_entropy_concept_x_context)
                                            / c_entropy_concept_x_context)

    # normalized version of h(m|c), i.e. h(m|c)/h(m)
    normalized_consistency = (joint_mc_entropy - c_entropy) / m_entropy
    normalized_consistency_hierarchical = (joint_entropy_hierarchical - c_entropy_hierarchical) / m_entropy_hierarchical
    # print("cons hier", normalized_consistency_hierarchical)
    normalized_consistency_context_dep = (joint_entropy_context_dep - c_entropy_context_dep) / m_entropy_context_dep
    normalized_consistency_conc_x_cont = (
                                                     joint_entropy_concept_x_context - c_entropy_concept_x_context) / m_entropy_concept_x_context

    score_dict = {'normalized_mutual_info': normalized_MI,
                  'normalized_mutual_info_hierarchical': normalized_MI_hierarchical,
                  'normalized_mutual_info_context_dep': normalized_MI_context_dep,
                  'normalized_mutual_info_concept_x_context': normalized_MI_conc_x_cont,
                  'effectiveness': 1 - normalized_effectiveness,
                  'effectiveness_hierarchical': 1 - normalized_effectiveness_hierarchical,
                  'effectiveness_context_dep': 1 - normalized_effectiveness_context_dep,
                  'effectiveness_concept_x_context': 1 - normalized_effectiveness_conc_x_cont,
                  'consistency': 1 - normalized_consistency,
                  'consistency_hierarchical': 1 - normalized_consistency_hierarchical,
                  'consistency_context_dep': 1 - normalized_consistency_context_dep,
                  'consistency_concept_x_context': 1 - normalized_consistency_conc_x_cont
                  }
    return score_dict


def cooccurrence_per_hierarchy_level(interaction, n_attributes, n_values, vs_factor, is_gumbel=True, trim_eos=False):
    vocab_size = (n_values + 1) * vs_factor + 1

    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message.argmax(dim=2)
    if trim_eos:
        # trim message to first EOS symbol
        messages = [trim_tensor(message) for message in messages]
        # Pad tensors to make them uniform length
        padded_tensors = [F.pad(t, pad=(0, max_mess_len - t.size(0))) for t in messages]
        # Stack into a single tensor
        messages = torch.stack(padded_tensors)
    else:
        messages = messages[:, :-1].numpy()

    sender_input = interaction.sender_input.numpy()
    relevance_vectors = sender_input[:, -n_attributes:]

    cooccurrence = np.zeros((vocab_size, n_attributes))

    for s in range(vocab_size):
        for i, m in enumerate(messages):
            relevance = relevance_vectors[i]
            cooccurrence[s, int(sum(relevance))] += list(m).count(s)

    cooccurrence = cooccurrence[1:, :]  # remove eos symbol
    split_indices = np.array([np.sum(sender_input[:, -n_attributes:], axis=1) == i for i in range(n_attributes)])
    normalization = np.array([np.sum(split_indices[i]) for i in range(n_attributes)])
    cooccurrence = cooccurrence / normalization

    return cooccurrence


def message_length_per_hierarchy_level(interaction, n_attributes, is_gumbel=True):
    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)
    n_values = int(sender_input.shape[2] / n_attributes)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)

    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message.argmax(dim=2)
    ml = MessageLengthHierarchical.compute_message_length(messages)
    ml_hierarchical = MessageLengthHierarchical.compute_message_length_hierarchical(messages, torch.from_numpy(fixed))
    return (ml, ml_hierarchical)


def message_length_per_context_condition(interaction, n_attributes):
    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)
    n_values = int(sender_input.shape[2] / n_attributes)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    distractor_objects = distractor_objects + 1
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)

    message = interaction.message.argmax(dim=-1)
    ml = MessageLengthHierarchical.compute_message_length(message)
    ml_context, ml_fine_context, ml_coarse_context = MessageLengthHierarchical.compute_message_length_over_context(
        message, torch.from_numpy(fixed), torch.from_numpy(np.array(context_conds)))
    return ml_context, ml_fine_context, ml_coarse_context


def effective_vocab_size(interaction, vocab_size, is_gumbel=True):
    """
    determines effectively used symbols
    returns the effective vocab size, the counts for each symbol in vocab, and the ratio of effective vocab size by
    possible vocab size
    """
    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message.argmax(dim=2)
    messages = [msg.tolist() for msg in messages]
    all_symbols = [symbol for message in messages for symbol in message]
    symbol_counts = Counter(all_symbols)
    return len(symbol_counts), symbol_counts, len(symbol_counts)/vocab_size


def symbol_frequency(interaction, n_attributes, n_values, vocab_size, is_gumbel=True, trim_eos=False):
    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message.argmax(dim=2)
    if trim_eos:
        # trim message to first EOS symbol
        messages = [trim_tensor(message) for message in messages]
        # Pad tensors to make them uniform length
        padded_tensors = [F.pad(t, pad=(0, max_mess_len - t.size(0))) for t in messages]
        # Stack into a single tensor
        messages = torch.stack(padded_tensors)
    else:
        messages = messages[:, :-1] # excluding EOS symbol 0
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)
    # k_hots = sender_input[:, :-n_attributes]
    # objects = k_hot_to_attributes(k_hots, n_values)
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # intentions = sender_input[:, -n_attributes:]  # (0=same, 1=any)
    (objects, fixed) = retrieve_concepts_sampling(target_objects)

    objects[fixed == 1] = np.nan

    objects = objects
    messages = messages
    favorite_symbol = {}
    mutual_information = {}
    for att in range(n_attributes):
        for val in range(n_values):
            object_labels = (objects[:, att] == val).astype(int)
            max_MI = 0
            for i, symbol in enumerate(range(vocab_size)):
                symbol_indices = np.argwhere(messages == symbol)[0]
                symbol_labels = np.zeros(len(messages))
                symbol_labels[symbol_indices] = 1
                MI = normalized_mutual_info_score(symbol_labels, object_labels)
                if MI >= max_MI:
                    max_MI = MI
                    max_symbol = symbol
            favorite_symbol[str(att) + str(val)] = max_symbol
            mutual_information[str(att) + str(val)] = max_MI

    sorted_objects = []
    sorted_messages = []
    for i in reversed(range(n_attributes)):
        sorted_objects.append(objects[np.sum(np.isnan(objects), axis=1) == i])
        sorted_messages.append(messages[np.sum(np.isnan(objects), axis=1) == i])

    # from most concrete to most abstract (all same to only one same)
    att_val_frequency = np.zeros(n_attributes)
    symbol_frequency = np.zeros(n_attributes)

    for level in range(n_attributes):
        for obj, message in zip(sorted_objects[level], sorted_messages[level]):
            for position in range(len(obj)):
                if not np.isnan(obj[position]):
                    att_val_frequency[level] += 1
                    fav_symbol = favorite_symbol[str(position) + str(int(obj[position]))]
                    symbol_frequency[level] += np.count_nonzero(message == fav_symbol)

    return symbol_frequency / att_val_frequency, mutual_information
    
# the function symbol_frequency_MI copied from zero_shot repository, used in split_analysis_mi_scores.ipynb
def symbol_frequency_MI(interaction, n_attributes, n_values, vocab_size, is_gumbel=True):
    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message
    messages = messages[:, :-1]  # without EOS
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    (objects, fixed) = retrieve_concepts_sampling(target_objects)

    # attributes which are not fixed are irrelevant to the concept and do not need to be communicated
    objects[fixed == 0] = np.nan

    favorite_symbol_MI = {}
    mutual_information = {}
    # find symbol with highest MI for each value at each attribute position (i.e., position sensitive)
    for att in range(n_attributes):
        for val in range(n_values):
            object_labels = (objects[:, att] == val).astype(int)
            max_MI = 0
            for symbol in range(vocab_size):
                symbol_indices = np.argwhere(messages == symbol)[0]
                symbol_labels = np.zeros(len(messages))
                symbol_labels[symbol_indices] = 1
                MI = normalized_mutual_info_score(symbol_labels, object_labels)
                if MI > max_MI:
                    max_MI = MI
                    max_symbol = symbol
            favorite_symbol_MI[str(att) + str(val)] = max_symbol
            mutual_information[str(att) + str(val)] = max_MI

    return favorite_symbol_MI, mutual_information

# symbol_frequency_MI modified for length cost datasets, used in split_analysis_mi_scores.ipynb
# masks all symbols after the first zero in each message (turns them into zeros)
# only considers non-zero symbols
def symbol_frequency_MI_new(interaction, n_attributes, n_values, vocab_size, is_gumbel=True):
    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message
    messages = messages[:, :-1]  # without EOS
    # Mask out everything after the first zero in each row
    mask = (messages != 0).type(torch.int)
    mask = mask.cumprod(dim=1)  # stays 1 until first 0, then 0
    messages = messages * mask
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    (objects, fixed) = retrieve_concepts_sampling(target_objects)

    # attributes which are not fixed are irrelevant to the concept and do not need to be communicated
    objects[fixed == 0] = np.nan

    favorite_symbol_MI = {}
    mutual_information = {}
    # find symbol with highest MI for each value at each attribute position (i.e., position sensitive)
    for att in range(n_attributes):
        for val in range(n_values):
            object_labels = (objects[:, att] == val).astype(int)
            max_MI = 0
            # only looks at symbols > 0
            for symbol in range(1, vocab_size):
                symbol_indices = np.argwhere(messages == symbol)[0]
                symbol_labels = np.zeros(len(messages))
                symbol_labels[symbol_indices] = 1
                MI = normalized_mutual_info_score(symbol_labels, object_labels)
                if MI > max_MI:
                    max_MI = MI
                    max_symbol = symbol
            favorite_symbol_MI[str(att) + str(val)] = max_symbol
            mutual_information[str(att) + str(val)] = max_MI

    return favorite_symbol_MI, mutual_information

def symbol_frequency_fav(interaction, n_attributes, n_values, vocab_size, is_gumbel=True, trim_eos=False,
                       max_mess_len=None):
    """

    """
    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message.argmax(dim=2)
    if trim_eos:
        # trim message to first EOS symbol
        messages = [trim_tensor(message) for message in messages]
        # Pad tensors to make them uniform length
        padded_tensors = [F.pad(t, pad=(0, max_mess_len - t.size(0))) for t in messages]
        # Stack into a single tensor
        messages = torch.stack(padded_tensors)
    else:
        messages = messages[:, :-1]  # without EOS
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects / 2)
    # k_hots = sender_input[:, :-n_attributes]
    # objects = k_hot_to_attributes(k_hots, n_values)
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # intentions = sender_input[:, -n_attributes:]  # (0=same, 1=any)
    (objects, fixed) = retrieve_concepts_sampling(target_objects)

    objects[fixed == 1] = np.nan

    objects = objects
    messages = messages
    favorite_symbol = {}
    mutual_information = {}
    for att in range(n_attributes):
        for val in range(n_values):
            object_labels = (objects[:, att] == val).astype(int)
            max_MI = 0
            for symbol in range(1, vocab_size):
                symbol_indices = np.argwhere(messages == symbol)[0]
                symbol_labels = np.zeros(len(messages))
                symbol_labels[symbol_indices] = 1
                MI = normalized_mutual_info_score(symbol_labels, object_labels)
                if MI >= max_MI:
                    max_MI = MI
                    max_symbol = symbol
            favorite_symbol[str(att) + str(val)] = max_symbol
            mutual_information[str(att) + str(val)] = max_MI

    return favorite_symbol, mutual_information


def get_fixed_vectors(sender_input, n_values, idx):
    """retrieves concepts, i.e. objects and fixed vectors from a sender input"""
    # obtain total counts of specific - generic concepts and fine - coarse contexts in dataset
    n_targets = int(sender_input[idx].shape[0] / 2)
    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[idx][:n_targets]
    target_objects = k_hot_to_attributes(target_objects.unsqueeze(0), n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects, all_targets=True)
    return (objects, fixed)


def get_context_cond(sender_input, n_values, idx, objects, fixed):
    """retrieves context condition from a sender input,
    needs objects and fixed which are calculated with get_fixed_vectors from the sender input"""
    n_targets = int(sender_input[idx].shape[0] / 2)
    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[idx][n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects.unsqueeze(0), n_values)
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)
    return context_conds


def obtain_concept_counts(sender_input, n_values):
    """calculates how many times a concept was shown from a sender input"""
    concepts = {}
    for idx in range(len(sender_input)):
        (objects, fixed) = get_fixed_vectors(sender_input, n_values, idx)
        concept_str = str(int(sum(fixed[0])))
        if concept_str in concepts:
            concepts[concept_str] += 1
        else:
            concepts[concept_str] = 1
    return concepts


def obtain_context_counts(sender_input, n_values):
    """calculates how many times a context condition was shown from a sender input"""
    contexts = {}
    for idx in range(len(sender_input)):
        # need concepts to calculate context conditions
        (objects, fixed) = get_fixed_vectors(sender_input, n_values, idx)
        context_conds = get_context_cond(sender_input, n_values, idx, objects, fixed)
        context_str = str(context_conds[0])
        if context_str in contexts:
            contexts[context_str] += 1
        else:
            contexts[context_str] = 1
    return contexts


def obtain_concept_x_context_counts(sender_input, n_values):
    """calculates how many concepts are in each condition from a sender input"""
    concept_x_context = {}
    for idx in range(len(sender_input)):
        (objects, fixed) = get_fixed_vectors(sender_input, n_values, idx)
        context_conds = get_context_cond(sender_input, n_values, idx, objects, fixed)
        concept_x_context_str = (context_conds[0], int(sum(fixed[0]) - 1))
        if concept_x_context_str in concept_x_context:
            concept_x_context[concept_x_context_str] += 1
        else:
            concept_x_context[concept_x_context_str] = 1
    return concept_x_context


def error_analysis(datasets, paths, setting, n_epochs, n_values, validation=True):
    """
    goes through interactions, retrieves concept and context conditions and counts errors,
    i.e. incorrectly classified objects (by the receiver)
    """
    all_error_concepts = {}
    all_error_contexts = {}
    all_error_concept_x_context = {}

    all_acc_concept_x_context = {}

    all_total_concepts = {}
    all_total_contexts = {}
    all_total_concept_x_context = {}

    # go through all datasets
    for i, d in enumerate(datasets):
        print(i, d)
        error_concepts = {}
        error_contexts = {}
        error_concept_x_context = {}
        acc_concept_x_context = {}
        # select first run
        path_to_run = paths[i] + '/' + str(setting) + '/' + str(0) + '/'
        path_to_interaction_train = (path_to_run + 'interactions/train/epoch_' + str(n_epochs) + '/interaction_gpu0')
        path_to_interaction_val = (path_to_run + 'interactions/validation/epoch_' + str(n_epochs) + '/interaction_gpu0')
        if validation:
            interaction = torch.load(path_to_interaction_val)
        else:
            interaction = torch.load(path_to_interaction_train)

        total_concepts = obtain_concept_counts(interaction.sender_input, n_values[i])

        total_contexts = obtain_context_counts(interaction.sender_input, n_values[i])

        total_concept_x_context = obtain_concept_x_context_counts(interaction.sender_input, n_values[i])

        for j in range(len(interaction.sender_input)):
            receiver_pred = (interaction.receiver_output[j][-1] > 0).float()  # use last symbol of message

            (objects, fixed) = get_fixed_vectors(interaction.sender_input, n_values[i], j)
            concept_str = str(int(sum(fixed[0])))

            context_conds = get_context_cond(interaction.sender_input, n_values[i], j, objects, fixed)
            context_str = str(context_conds[0])
            concept_x_context_str = (context_conds[0], int(sum(fixed[0]) - 1))

            # check if receiver has classified all objects correctly as targets or distractors
            if not torch.equal(receiver_pred, interaction.labels[j]):
                if concept_str in error_concepts:
                    error_concepts[concept_str] += 1
                else:
                    error_concepts[concept_str] = 1

                if context_str in error_contexts:
                    error_contexts[context_str] += 1
                else:
                    error_contexts[context_str] = 1

                if concept_x_context_str in error_concept_x_context:
                    error_concept_x_context[concept_x_context_str] += 1
                else:
                    error_concept_x_context[concept_x_context_str] = 1

            # check if receiver has classified some objects correctly as targets or distractors
            # (this is how the accuracy is calculated during training)
            if concept_x_context_str in acc_concept_x_context:
                acc_concept_x_context[concept_x_context_str] += (
                    (receiver_pred == interaction.labels[j]).float().mean().numpy())
            else:
                acc_concept_x_context[concept_x_context_str] = (
                        receiver_pred == interaction.labels[j]).float().mean().numpy()

        all_error_concepts[d] = error_concepts
        all_error_contexts[d] = error_contexts
        all_error_concept_x_context[d] = error_concept_x_context

        all_acc_concept_x_context[d] = acc_concept_x_context

        all_total_concepts[d] = total_concepts
        all_total_contexts[d] = total_contexts
        all_total_concept_x_context[d] = total_concept_x_context

    return (all_error_concepts, all_error_contexts, all_error_concept_x_context, all_acc_concept_x_context,
            all_total_concepts, all_total_contexts, all_total_concept_x_context)


def informativeness_score(interaction, distance='manhattan'):
    """
    Takes a distance function.
    Calculates informativeness of a lexicon based on the measure proposed by Gualdoni et al. (2024).
    """
    if distance not in ['manhattan', 'euclidean']:
        print("Distance must be either 'manhattan' or 'euclidean'.")
        return

    _, counts_sender_input = torch.unique(interaction.sender_input, return_counts=True, dim=0)
    unique_messages, message_indices, message_counts = torch.unique(interaction.message, return_inverse=True,
                                                                    return_counts=True, dim=0)
    # TODO: find effective message length and use messages according to effective message length?
    informativeness = []
    # for each unique message:
    for i, m in enumerate(unique_messages):
        # find out all referents
        referents = []
        for j, idx in enumerate(message_indices):
            if idx == i:
                # subset :10 to only consider targets, not distractors
                referents.append(interaction.sender_input[j][:10])
        if len(referents) > 1:
            # compute pairwise distances between referents
            # (e.g. (_,2,1) is more similar to (_,1,1) than to (_,1,3) or to (_,_,3))
            distances = []
            tmp = []
            for k, ref in enumerate(referents):
                for j in range(len(referents)):
                    if k != j:
                        if distance == 'manhattan':
                            dist = torch.sum(torch.abs(ref - referents[j]))
                            distances.append(dist)
                        elif distance == 'euclidean':
                            dist = torch.norm(ref - referents[j])
                            distances.append(dist)
            distances = [d for d in distances if d != 0.0]
            if len(distances) != 0:
                tmp.append(sum(distances) / len(distances))
            tmp = [t for t in tmp if t != 0.0]
            # word informativeness is 1/averaged pairwise distances, *10^2 as in original paper for better visibility
            if len(tmp) != 0:
                informativeness.append(100 / (sum(tmp) / len(tmp)))
    # informativeness of a lexicon is word informativeness per word averaged
    lex_info = np.sum(informativeness) / len(informativeness)
    return lex_info, len(unique_messages), len(counts_sender_input)
