# code inspired by https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/dataset.py

import torch
import torch.nn.functional as F
import itertools
import random
from tqdm import tqdm

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)


class DataSet(torch.utils.data.Dataset):
    """
    This class provides the torch.Dataloader-loadable dataset.
    """

    def __init__(self, properties_dim=None, game_size=10, device=None, testing=False, zero_shot=False, zero_shot_test=None):
        """
		properties_dim: vector that defines how many attributes and features per attributes the dataset should contain, defaults to a 3x3x3 dataset
		game_size: integer that defines how many targets and distractors a game consists of
		"""
	    if properties_dim is None:
			properties_dim = [3, 3, 3]
		if device is None:
			device =  'cuda'
		if zero_shot_test is None:
			zero_shot_test = 'generic'

        super().__init__()

        self.properties_dim = properties_dim
        self.game_size = game_size
        self.device = device

        # get all concepts
        self.concepts = self.get_all_concepts()
        # get all objects
        self.all_objects = self._get_all_possible_objects(properties_dim)

        # generate dataset
        if not testing and not zero_shot:
            self.dataset = self.get_datasets(split_ratio=SPLIT)
        if zero_shot:
            self.dataset = self.get_zero_shot_datasets(
                split_ratio=SPLIT_ZERO_SHOT, test_cond=zero_shot_test
            )

    def __len__(self):
        """Returns the total amount of samples in dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns the i-th sample (and label?) given an index (idx)."""
        return self.dataset[idx]

    def get_datasets(self, split_ratio, include_concept=False):
        """
        Creates the train, validation and test datasets based on the number of possible concepts.
        """
        if sum(split_ratio) != 1:
            raise ValueError

        train_ratio, val_ratio, test_ratio = split_ratio

        # Shuffle sender indices
        concept_indices = torch.randperm(len(self.concepts)).tolist()
        # Split is based on how many distinct concepts there are (regardless context conditions)
        ratio = int(len(self.concepts) * (train_ratio + val_ratio))

        train_and_val = []
        print("Creating train_ds and val_ds...")
        for concept_idx in tqdm(concept_indices[:ratio]):
            for _ in range(self.game_size):
                # for each concept, we consider all possible context conditions
                # i.e. 1 for generic concepts, and up to len(properties_dim) for more specific concepts
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                for context_condition in range(nr_possible_contexts):
                    train_and_val.append(
                        self.get_item(
                            concept_idx,
                            context_condition,
                            self._normalized_encoding,
                            include_concept,
                        )
                    )

        # Calculating how many train
        train_samples = int(
            len(train_and_val) * (train_ratio / (train_ratio + val_ratio))
        )
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(
            train_and_val, [train_samples, val_samples]
        )
        # Save information about train dataset
        train.dimensions = self.properties_dim

        test = []
        print("\nCreating test_ds...")
        for concept_idx in tqdm(concept_indices[ratio:]):
            for _ in range(self.game_size):
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                for context_condition in range(nr_possible_contexts):
                    test.append(
                        self.get_item(
                            concept_idx,
                            context_condition,
                            self._normalized_encoding,
                            include_concept,
                        )
                    )

        return train, val, test

    def get_zero_shot_datasets(
        self, split_ratio, test_cond="generic", include_concept=False
    ):
        """
        Note: Generates train, val and test data.
            Test and training set contain different concepts. There are two possible datasets:
            1) 'generic': train on more specific concepts, test on most generic concepts
            2) 'specific': train on more generic concepts, test on most specific concepts
        :param split_ratio Tuple of ratios (train, val) of the samples should be in the training and validation sets.
        """

        if sum(split_ratio) != 1:
            raise ValueError

            # For each category, one attribute will be chosen for zero shot
            # The attributes will be taken from a random object
            # zero_shot_object = pd.Series([0 for _ in self.properties_dim])  # self.objects.sample().iloc[0]

            # split ratio applies only to train and validation datasets - size of test dataset depends on available concepts
        train_ratio, val_ratio = split_ratio

        train_and_val = []
        test = []

        print("Creating train_ds, val_ds and test_ds...")
        for concept_idx in tqdm(range(len(self.concepts))):
            for _ in range(self.game_size):
                # for each concept, we consider all possible context conditions
                # i.e. 1 for generic concepts, and up to len(properties_dim) for more specific concepts
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                # print("nr poss cont", nr_possible_contexts)
                for context_condition in range(nr_possible_contexts):
                    # 1) 'generic'
                    if test_cond == "generic":
                        # test dataset only contains most generic concepts
                        if nr_possible_contexts == 1:
                            test.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._normalized_encoding,
                                    include_concept,
                                )
                            )
                        else:
                            train_and_val.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._normalized_encoding,
                                    include_concept,
                                )
                            )

                            # 2) 'specific'
                    if test_cond == "specific":
                        # test dataset only contains most specific concepts
                        if nr_possible_contexts == len(self.properties_dim):
                            test.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._normalized_encoding,
                                    include_concept,
                                )
                            )
                        else:
                            train_and_val.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._normalized_encoding,
                                    include_concept,
                                )
                            )

                            # Train val split
        train_samples = int(len(train_and_val) * train_ratio)
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(
            train_and_val, [train_samples, val_samples]
        )

        # Save information about train dataset
        train.dimensions = self.properties_dim
        print("Length of train and validation datasets:", len(train), "/", len(val))
        print("Length of test dataset:", len(test))

        return train, val, test

    def get_item(
        self, concept_idx, context_condition, encoding_func, include_concept=False
    ):
        """
        Receives concept-context pairs and an encoding function.
        Returns encoded (sender_input, labels, receiver_input).
                sender_input: (sender_input_objects, sender_labels)
                labels: indices of target objects in the receiver_input
                receiver_input: receiver_input_objects
        The sender_input_objects and the receiver_input_objects are different objects sampled from the same concept
        and context condition.
        """
        # use get_sample() to get sampled target and distractor objects
        # The concrete sampled objects can differ between sender and receiver.
        sender_concept, sender_context = self.get_sample(concept_idx, context_condition)
        receiver_concept, receiver_context = self.get_sample(
            concept_idx, context_condition
        )
        # TODO: change such that sender input also includes fixed vectors (i.e. full concepts) and fixed vectors are only
        # ignored in the sender architecture
        # NOTE: also do this for context conditions?
        # initalize sender and receiver input with target objects only
        if include_concept == True:
            raise NotImplementedError

        # subset such that only target objects are presented to sender and receiver
        sender_targets = sender_concept[0]
        receiver_targets = receiver_concept[0]
        sender_input = [obj for obj in sender_targets]
        receiver_input = [obj for obj in receiver_targets]
        # append context objects
        # get context of relevant context condition
        for distractor_objects, context_cond in sender_context:
            if context_cond == context_condition:
                # add distractor objects for the sender
                for obj in distractor_objects:
                    sender_input.append(obj)
        for distractor_objects, context_cond in receiver_context:
            if context_cond == context_condition:
                # add distractor objects for the receiver
                for obj in distractor_objects:
                    receiver_input.append(obj)
        # sender input does not need to be shuffled - that way I don't need labels either
        # random.shuffle(sender_input)
        # sender_label = [idx for idx, obj in enumerate(sender_input) if obj in sender_targets]
        # sender_label = torch.Tensor(sender_label).to(torch.int64)
        # sender_label = F.one_hot(sender_label, num_classes=self.game_size*2).sum(dim=0).float()
        # shuffle receiver input and create (many-hot encoded) label
        random.shuffle(receiver_input)
        receiver_label = [
            idx for idx, obj in enumerate(receiver_input) if obj in receiver_targets
        ]
        receiver_label = (
            torch.Tensor(receiver_label).to(torch.int64).to(device=self.device)
        )
        receiver_label = (
            F.one_hot(receiver_label, num_classes=self.game_size * 2).sum(dim=0).float()
        )
        # ENCODE and return as TENSOR
        sender_input = torch.stack([encoding_func(elem) for elem in sender_input])
        receiver_input = torch.stack([encoding_func(elem) for elem in receiver_input])
        # output needs to have the structure sender_input, labels, receiver_input
        # return torch.cat([sender_input, sender_label]), receiver_label, receiver_input
        return sender_input, receiver_label, receiver_input

    def get_sample(self, concept_idx, context_condition):
        """
        Returns a full sample consisting of a set of target objects (target concept)
        and a set of distractor objects (context) for a given concept condition.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        # Define the range within which an object's attribute value can vary to be considered a target
        range_threshold = 0.1  # This threshold can be adjusted as needed

        # Sample target objects within the specified range threshold
        target_objects = [
            obj for obj in all_target_objects if all(
                abs(obj[i] - all_target_objects[0][i]) < range_threshold
                for i, fix in enumerate(fixed) if fix == 1
            )
        ]

        # If the number of target objects is less than the game size, sample with replacement
        if len(target_objects) < self.game_size:
            target_objects = random.choices(target_objects, k=self.game_size)
        else:
            target_objects = random.sample(target_objects, self.game_size)

        # Get all possible distractors for a given concept (for all possible context conditions).
        context = self.get_distractors(concept_idx, context_condition)
        context_sampled = self.sample_distractors(context, context_condition)

        return [target_objects, fixed], context_sampled

    def get_distractors(self, concept_idx):
        """
        Computes distractors for concepts with continuous attributes. Distractors are based purely on 
        the similarity score, not on a discrete count of matching attributes like in orginal dataset.
        Therefore context_condition is not used as an argument 
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        context = []
        similarity_threshold = 0.1  # Define how close a distractor's attributes should be to the target's.
        for obj in self.all_objects:
            similarity_score = self.compute_similarity(obj, all_target_objects[0], fixed)
            if similarity_score >= similarity_threshold:
                context.append(obj)
        return context

    def compute_similarity(self, object1, object2, fixed):
        """
        Computes a similarity score based on continuous attributes.
        """
        score = 0
        for i, is_fixed in enumerate(fixed):
            if is_fixed == 1:
                score += 1 - abs(object1[i] - object2[i]) 
        # Normalize the score by the number of fixed attributes, if not zero.
        return score / sum(fixed) if sum(fixed) != 0 else 0

    def sample_distractors(self, context, context_condition):
        """
        Function for sampling the distractors from a specified context condition.
        """
        # sample distractor objects for given game size and the specified context condition
        # distractors = [dist_obj for dist_objs in context for dist_obj in dist_objs]
        context_new = []
        try:
            context_new.append(
                [random.sample(context, self.game_size), context_condition]
            )
        except ValueError:
            context_new.append(
                [random.choices(context, k=self.game_size), context_condition]
            )
        return context_new

    def sample_distractors_old(self, distractors, fixed):
        """
        Function for sampling the distractors from all possible context conditions.
        """
        # sample distractor objects for given game size and each context condition (constrained by level of abstraction)
        context = list()
        context_candidates = list()
        for i in range(sum(fixed)):
            for dist_objects, context_condition in distractors:
                # check for context condition
                # sum(context_condition) gives the number of shared attributes
                if sum(context_condition) == i:
                    for dist_object in dist_objects:
                        context_candidates.append([dist_object, i])
        helper_i = 0
        helper_list = list()
        # for i in range(len(self.properties_dim)):
        for i, (dist_object, context_condition) in enumerate(context_candidates):
            if helper_i == context_condition:
                # gather all objects belonging to the same context condition
                helper_list.append(dist_object)
                # final index: should be sampled
                if i == len(context_candidates) - 1:
                    try:
                        context.append(
                            [random.sample(helper_list, self.game_size), helper_i]
                        )
                    except ValueError:
                        context.append(
                            [random.choices(helper_list, k=self.game_size), helper_i]
                        )
            # catch the final context condition as well
            elif context_condition == len(self.properties_dim) - 1:
                try:
                    context.append(
                        [random.sample(helper_list, self.game_size), helper_i]
                    )
                except ValueError:
                    context.append(
                        [random.choices(helper_list, k=self.game_size), helper_i]
                    )
                helper_i = helper_i + 1
                helper_list = list()
                helper_list.append(dist_object)
            # when moving to the next context condition, first sample from the old
            else:
                # sample from all objects belonging to the same context condition
                try:
                    context.append(
                        [random.sample(helper_list, self.game_size), helper_i]
                    )
                except ValueError:
                    context.append(
                        [random.choices(helper_list, k=self.game_size), helper_i]
                    )
                helper_i = helper_i + 1
                helper_list = list()
                helper_list.append(dist_object)
        return context

    def get_all_concepts(self):
        """
        Returns all possible concepts for a given dataset size with continuos attributes.
        Each concept is represented as a range of values for each attribute. 
        """
        all_objects = self._get_all_possible_objects(self.properties_dim)
        concepts = []
        # Define the width of the range around the attribute value that is considered part of the concept.
        range_width = 0.1
        
        for obj in all_objects:
            concept = []  # Initialize an empty list to represent a single concept.
            for i, value in enumerate(obj):  
                # Create a range around each attribute value.
                attribute_range = (max(0, value - range_width), min(1, value + range_width))
                concept.append(attribute_range)  
            concepts.append(tuple(concept))  # Add the concept as a tuple to the list of concepts.
        return concepts 
           

    def get_shared_vectors(self, fixed):
        """
        Returns fixed vectors for all possible context conditions based on a concept (i.e. the fixed vector).
        These are called "shared_vectors" because the number and position of attributes which are shared with the
        target concept define the context condition. The more fixed attributes are shared, the finer the context.
        """
        shared_vectors = []
        for i, attribute in enumerate(fixed):
            shared = list(itertools.repeat(0, len(fixed)))
            if attribute == 1:
                shared[i] = 1
                shared_vectors.append(shared)
        return shared_vectors
    
    @staticmethod
    def satisfies(object, concept, threshold=0.1):
        """
        Checks whether an object satisfies a target concept, returns a boolean value.
        Concept consists of an object vector and a fixed vector tuple.
        A threshold is used to determine if a float attribute value matches.
        """
        concept_object, fixed = concept
        # Iterate through each attribute to check if it's fixed and within the threshold of the concept.
        for i, attr in enumerate(fixed):
            if attr == 1: #if attribute is fixed
                # Compare the object's attribute with the concept's
                if not (abs(object[i] - concept_object[i]) < threshold):
                    # If the attribute is not within the threshold, return False
                    return False
        #If the loop completes without returning False, all fixed attributes are within the threshold
        return True

    @staticmethod
    def get_fixed_vectors(properties_dim):
        """
        Returns all possible fixed vectors for a given dataset size.
        Fixed vectors are vectors of length len(properties_dim), where 1 denotes that an attribute is fixed, 0 that it isn't.
        The more attributes are fixed, the more specific the concept -- the less attributes fixed, the more generic the concept.
        """
        # what I want to get: [(1,0,0), (0,1,0), (0,0,1)] for most generic
        # concrete: [(1,1,0), (0,1,1), (1,0,1)]
        # most concrete: [(1,1,1)]
        # for variable dataset sizes

        # range(0,2) because I want [0,1] values for whether an attribute is fixed or not
        list_of_dim = [range(0, 2) for dim in properties_dim]
        fixed_vectors = list(itertools.product(*list_of_dim))
        # remove first element (0,..,0) as one attribute always has to be fixed
        fixed_vectors.pop(0)
        return fixed_vectors

    @staticmethod
    def get_all_objects_for_a_concept(properties_dim, feature_ranges):
        """
        Generates all possible objects for a concept given the range of each attribute for a dataset with continuous attributes.
        
        properties_dim: A list of the number of divisions or samples within the range for each attribute.
        feature_ranges: A list of tuples where each tuple consists of (min_value, max_value) for each attribute.
        
        Returns a list of concept objects where each object is a tuple of values within the specified ranges.
        """
        all_objects = []
        
        # For each attribute's range, generate a set of points.
        for dim, (min_val, max_val) in zip(properties_dim, feature_ranges):
            # Assumes an equal distribution of points within the range
            all_objects.append(torch.linspace(min_val, max_val, steps=dim).tolist())
            
        concept_objects = list(itertools.product(*all_objects))

        return concept_objects

    @staticmethod
    def _get_all_possible_objects(properties_dim):
        """
        Returns all possible combinations of attribute-feature values as a list of tuples.
        """
        # Create a list of torch tensors, each with randomly sampled values between 0 and 1
        continuous_dims = [torch.rand(dim) for dim in properties_dim]
    
        # Convert each tensor to a list and then create a Cartesian product of these lists
        all_objects = list(itertools.product(*[dim.tolist() for dim in continuous_dims]))
    
        return all_objects 

    def _normalized_encoding(self, input_list):
        """
        This function outputs a normalized encoded tensor for a given input list where each element in the input list corresponds to an attribute of an object, 
        and the values are indices within their respective attribute's range. 
        """
        output = torch.tensor(input_list, device=self.device)
        # Add a small epsilon to avoid division by zero
        output_sum = output.sum().item()
        if output_sum == 0:
            output_sum = 1e-10  # small number to avoid division by zero
        # Normalizes the vector to have a sum of 1
        output = output / output_sum
        return output
        


def get_distractors_old(self, concept_idx):
    """
    Returns all possible distractor objects for each context based on a given target concept.
    return (context, distractor_objects) tuples
    """

    target_objects, fixed = self.concepts[concept_idx]
    fixed = list(fixed)

    def change_one_attribute(input_object, fixed):
        """
        Returns a concept where one attribute is changed.
        Input: A concept consisting of an (example) object and a fixed vector indicating which attributes are fixed in the concept.
        Output: A list of concepts consisting of an (example) object that differs in one attribute from the input object and a new fixed vector.
        """
        changed_concepts = []
        # go through target object and fixed
        # O(n_attributes)
        for i, attribute in enumerate(input_object):
            # check whether attribute in target object is fixed
            if fixed[i] == 1:
                # change one attribute to all possible attributes that don't match the target_object
                # O(n_values)
                for poss_attribute in range(self.properties_dim[i]):
                    # new_fixed = fixed.copy() # change proposed by ChatGPT
                    if poss_attribute != attribute:
                        new_fixed = fixed.copy()  # change proposed by ChatGPT
                        new_fixed[i] = 0
                        changed = list(input_object)
                        changed[i] = poss_attribute
                        # the new fixed values specify where the change took place: (1,1,0) means the change took place in 3rd attribute
                        changed_concepts.append((changed, new_fixed))
        return changed_concepts

    def change_n_attributes(input_object, fixed, n_attributes):
        """
        Changes a given number of attributes from a target object
                given a fixed vector (specifiying the attributes that can and should be changed)
                and a target object
                and a number of how many attributes should be changed.
        """
        changed_concepts = list()
        # O(n_attributes),
        while n_attributes > 0:
            # if changed_concepts is empty, I consider the target_object
            if not changed_concepts:
                changed_concepts = [change_one_attribute(input_object, fixed)]
                n_attributes = n_attributes - 1
            # otherwise consider the changed concepts and change them again	 until n_attributes = 0
            else:
                old_changed_concepts = changed_concepts.copy()
                # O(game_size)
                for sublist in changed_concepts:
                    for changed_concept, fixed in sublist:
                        new_changed_concepts = change_one_attribute(
                            changed_concept, fixed
                        )
                        if new_changed_concepts not in old_changed_concepts:
                            old_changed_concepts.append(new_changed_concepts)
                # copy and store for next iteration
                changed_concepts = old_changed_concepts.copy()
                n_attributes = n_attributes - 1
        # flatten list
        changed_concepts_flattened = [
            changed_concept
            for sublist in changed_concepts
            for changed_concept in sublist
        ]
        # remove doubles
        changed_concepts_final = []
        [
            changed_concepts_final.append(x)
            for x in changed_concepts_flattened
            if x not in changed_concepts_final
        ]
        return changed_concepts_final

    # distractors: number and position of fixed attributes match target concept
    # the more fixed attributes are shared, the finer the context
    distractor_concepts = change_n_attributes(target_objects[0], fixed, sum(fixed))
    # the fixed vectors in the distractor_concepts indicate the number of shared features: (1,0,0) means only first attribute is shared
    # thus sum(fixed) indicates the context condition: from 0 = coarse to n_attributes = fine
    # for the dataset I need objects instead of concepts
    distractor_objects = []
    for dist_concept in distractor_concepts:
        # same fixed vector as for the target concept
        distractor_objects.extend(
            [
                (
                    self.get_all_objects_for_a_concept(
                        self.properties_dim, dist_concept[0], fixed
                    ),
                    tuple(dist_concept[1]),
                )
            ]
        )

    return distractor_objects
