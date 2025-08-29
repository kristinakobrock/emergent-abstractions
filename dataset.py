# code inspired by https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/dataset.py

import torch
import torch.nn.functional as F
import itertools
import random
from tqdm import tqdm

import numpy as np

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)


class DataSet(torch.utils.data.Dataset):
    """
    This class provides the torch.Dataloader-loadable dataset.
    """

    def __init__(self, properties_dim=[3, 3, 3], game_size=10, scaling_factor=10, device='cuda', testing=False,
                 zero_shot=False, zero_shot_test=None, sample_context=False, granularity="mixed", is_shapes3d=False,
                 images=[], labels=[], shared_context=False, split_by_attribute=False, percentage_a=0.8):
        """
        properties_dim: vector that defines how many attributes and features per attributes the dataset should contain,
        defaults to a 3x3x3 dataset
        game_size: integer that defines how many targets and distractors a game consists of
        split_by_attribute: new parameter that creates a dataset, where one attribute has higher 
        discriminative value than others.
        percentage_a: use for split_by_attribute; variable to manipulate the proporion of subset A in the train+val dataset. default: 80%
        """
        super().__init__()

        self.properties_dim = properties_dim
        self.game_size = game_size
        self.scaling_factor = scaling_factor
        self.device = device
        self.sample_context = sample_context
        self.granularity = granularity
        self.shared_context = shared_context
        self.split_by_attribute = split_by_attribute
        self.percentage_a = percentage_a

        # check if granularity has one of the allowed values
        if granularity not in ["mixed", "fine", "coarse"]:
            raise ValueError("Granularity must be one of: 'mixed', 'fine', or 'coarse'.")

        # check that sample context is not used together with fine or coarse granularities
        if sample_context and granularity in ["fine", "coarse"]:
            raise ValueError("Sample context can only be applied in the mixed granularity (standard) condition.")

        # if the flag for shapes3d is True the dataset will be created with the images and labels of shapes3d
        self.is_shapes3d = is_shapes3d

        if self.is_shapes3d:
            # images can also be feature representations
            self.images = images
            self.labels = labels
            self.properties_dim = [4, 4, 4]
            self.all_objects = list(set(self.reverse_one_hot()))
            self.encoding_func = self._sample_image_from_concept
        else:
            self.properties_dim = properties_dim
            self.all_objects = self._get_all_possible_objects(properties_dim)
            self.encoding_func = self._many_hot_encoding
        # get all concepts
        self.concepts = self.get_all_concepts()

        """ 
        for split_by_attribute:
        randomly choose one attribute which will be highly important 
        for discriminating between targets and distractors in the subset A.
        """ 
        self.discriminative_attribute = random.choice(range(len(self.properties_dim)))

        # generate dataset
        if not testing and not zero_shot and not split_by_attribute:
            self.dataset = self.get_datasets(split_ratio=SPLIT)
        elif zero_shot:
            # check if zero_shot_test has one of the allowed values
            if zero_shot_test not in ["specific", "generic", None]:
                raise ValueError("zero_shot_test is", zero_shot_test, "but must be either 'specific' or 'generic'.")
            if split_by_attribute:
                raise ValueError("zero_shot is incompatible with split_by_attribute.")
            self.dataset = self.get_zero_shot_datasets(split_ratio=SPLIT_ZERO_SHOT, test_cond=zero_shot_test)
        elif split_by_attribute:
            if not shared_context:
                raise ValueError("split_by_attribute requires shared_context to be true.")
            if granularity != "mixed":
                raise ValueError("split_by_attribute requires mixed granularity.")
            self.dataset = self.get_split_by_attribute(self.percentage_a)

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
            for _ in range(self.scaling_factor):
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                # for each concept, we can consider all possible context conditions -> "mixed" granularity
                # i.e. 1 for generic concepts, and up to len(properties_dim) for more specific concepts
                if not self.sample_context:
                    # standard: take all possible context conditions
                    if self.granularity == "mixed":
                        for context_condition in range(nr_possible_contexts):
                            train_and_val.append(
                                self.get_item(concept_idx, context_condition, self.encoding_func, include_concept))
                    # fine context condition has n_fixed-1 shared attributes between targets and distractors
                    # n.b. the non-shared attributes is *not* fixed
                    elif self.granularity == "fine":
                        train_and_val.append(
                            self.get_item(concept_idx, nr_possible_contexts - 1, self.encoding_func,
                                          include_concept))
                    # coarse context condition has no shared attributes between targets and distractors
                    elif self.granularity == "coarse":
                        train_and_val.append(
                            self.get_item(concept_idx, 0, self.encoding_func, include_concept))

                # or sample context condition from possible context conditions
                else:
                    context_condition = random.choice(range(nr_possible_contexts))
                    train_and_val.append(
                        self.get_item(concept_idx, context_condition, self.encoding_func, include_concept))

        # Calculating how many train
        train_samples = int(len(train_and_val) * (train_ratio / (train_ratio + val_ratio)))
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])
        # Save information about train dataset
        train.dimensions = self.properties_dim

        test = []
        print("\nCreating test_ds...")
        for concept_idx in tqdm(concept_indices[ratio:]):
            for _ in range(self.scaling_factor):
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                if not self.sample_context:
                    for context_condition in range(nr_possible_contexts):
                        test.append(
                            self.get_item(concept_idx, context_condition, self.encoding_func, include_concept))
                # or sample context condition from possible context conditions
                else:
                    context_condition = random.choice(range(nr_possible_contexts))
                    test.append(self.get_item(concept_idx, context_condition, self.encoding_func, include_concept))

        return train, val, test

    def get_zero_shot_datasets(self, split_ratio, test_cond='generic', include_concept=False):
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
            for _ in range(self.scaling_factor):
                # for each concept, we can consider all possible context conditions -> "mixed" granularity
                # i.e. 1 for generic concepts, and up to len(properties_dim) for more specific concepts
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                # standard: take all possible context conditions
                if self.granularity == "mixed" and not self.sample_context:
                    for context_condition in range(nr_possible_contexts):
                        # 1) 'generic'
                        if test_cond == 'generic':
                            # test dataset only contains most generic concepts
                            if nr_possible_contexts == 1:
                                assert context_condition == 0, (f'generic concepts only in coarse contexts but is '
                                                                f'{context_condition}')
                                test.append(
                                    self.get_item(concept_idx, context_condition, self.encoding_func,
                                                  include_concept))
                            else:
                                train_and_val.append(
                                    self.get_item(concept_idx, context_condition, self.encoding_func,
                                                  include_concept))

                        # 2) 'specific'
                        elif test_cond == 'specific':
                            # test dataset only contains most specific concepts
                            if nr_possible_contexts == len(self.properties_dim):
                                test.append(
                                    self.get_item(concept_idx, context_condition, self.encoding_func,
                                                  include_concept))
                            else:
                                train_and_val.append(
                                    self.get_item(concept_idx, context_condition, self.encoding_func,
                                                  include_concept))

                # fine contexts only:
                elif self.granularity == "fine":
                    context_condition = nr_possible_contexts - 1
                elif self.granularity == "coarse":
                    context_condition = 0
                elif self.sample_context and self.granularity == "mixed":
                    context_condition = random.choice(range(nr_possible_contexts))

                if not (self.granularity == "mixed" and not self.sample_context):
                    # 1) 'generic'
                    if test_cond == 'generic':
                        # test dataset only contains most generic concepts
                        if nr_possible_contexts == 1:
                            context_condition = 0  # for generic concepts, only coarse context condition exists
                            test.append(
                                self.get_item(concept_idx, context_condition, self.encoding_func,
                                              include_concept))
                        else:
                            train_and_val.append(
                                self.get_item(concept_idx, context_condition, self.encoding_func,
                                              include_concept))

                    # 2) 'specific'
                    elif test_cond == 'specific':
                        # test dataset only contains most specific concepts
                        if nr_possible_contexts == len(self.properties_dim):
                            test.append(
                                self.get_item(concept_idx, context_condition, self.encoding_func,
                                              include_concept))
                        else:
                            train_and_val.append(
                                self.get_item(concept_idx, context_condition, self.encoding_func,
                                              include_concept))

        # Train val split
        train_samples = int(len(train_and_val) * train_ratio)
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])

        # Save information about train dataset
        train.dimensions = self.properties_dim
        print("Length of train and validation datasets:", len(train), "/", len(val))
        print("Length of test dataset:", len(test))

        return train, val, test

    def get_split_by_attribute(self, percentage_a):
        """
        Creates two subsets, A and B, based on a randomly chosen attribute (discriminative_attribute).
        Requirements: 
            shared_context must be true. This function assumes it is true based on the check in init!
        Subset A: 
            Concept-context pairs where targets and distractors differ only in discriminative_attribute:
            discriminative_attribute is fixed, context is "fine", and the discriminative_attribute is NOT shared with distractors.
            This way, one randomly chosen attribute is highly important for discrimination between targets and distractors.
        Subset B:
            All other concept-context pairs, where discriminative_attribute is not important for
            discrimination beween targets and distractors.
        The function then splits these two subsets for training, validation, and testing (see comment below for splitting logic).
        """
        # subset A constitutes 0-100% of the train+val dataset
        if percentage_a > 1 or percentage_a < 0:
            raise ValueError("percentage_a must be between 0 and 1")

        subset_a=[] #concept-context pairs where one randomly chosen attribute is highly discriminative
        subset_b=[] #other concept-context pairs
        print(f"Creating subsets A and B based on randomly chosen attribute. Attribute index: {self.discriminative_attribute}")

        for concept_idx in tqdm(range(len(self.concepts))):
            target_objects, fixed = self.concepts[concept_idx]
            # the amount of possible context conditions is the amount of fixed attrbutes (e.g. 1 attribute fixed -> one condition possible)
            nr_possible_contexts = sum(fixed)

            # variable to check if the randomly chosen attribute X is fixed in the concept (returns True or False)
            attribute_is_fixed = fixed[self.discriminative_attribute] == 1

            # for subset A: choose the concept-context pairs, where 1) the randomly chosen attribute is fixed,
            # 2)context is shared (already assumed in init), 3) granularity is fine and 4) the chosen attribute is not shared with distractors
            for _ in range(self.scaling_factor):
                # check if chosen attribute is fixed
                if attribute_is_fixed:
                   for context_condition in range(nr_possible_contexts):
                       # choose the finest possible context condition, for example for 2-attribute concept it will be the condition "1"
                       if context_condition == nr_possible_contexts - 1: 
                          # call get_distractors_shared with split_by_attribute_subset_a=True to influence shared_attr_indices selection
                          _, shared_attr_indices = self.get_distractors_shared(concept_idx, context_condition, split_by_attribute_subset_a=True)
                          # check if the discriminative attribute is NOT in the shared attributes for subset A
                          if self.discriminative_attribute not in shared_attr_indices:
                              subset_a.append(self.get_item(concept_idx, context_condition, self.encoding_func, include_concept=False))
                          #else:
                              #subset_b.append(self.get_item(concept_idx, context_condition, self.encoding_func, include_concept=False))
                       else:
                          # if context condition is not fine, add to subset B
                          subset_b.append(self.get_item(concept_idx, context_condition, self.encoding_func, include_concept=False))
                else:
                    # if the discriminative attribute is not fixed, add to subset B
                     for context_condition in range(nr_possible_contexts):
                         subset_b.append(
                             self.get_item(concept_idx, context_condition, self.encoding_func, include_concept=False))

        print(f"Subset A size: {len(subset_a)}")
        print(f"Subset B size: {len(subset_b)}")

        """
        Splitting logic.
        Now, subsets A and B are used to create train, val and test datasets.
        The whole subset A and a portion of subset B are used for training and validation.
        The parameter "percentage_a" determines the proportion of the train+val dataset that comes from subset A.
        For example, if percentage_a is 0.8, 80% of the train+val samples are taken from subset A,
        and the remaining 20% come from subset B.
        """

        # calculate number of samples from subset B that will be used for train+val
        number_b_for_train_val = int(len(subset_a)/percentage_a) - len(subset_a)
        # randomly sample the needed amount of concept-context pairs from subset B
        random.shuffle(subset_b)
        b_train_val = subset_b[:number_b_for_train_val]
        # remaining part of subset B is used for testing
        test = subset_b[number_b_for_train_val:]

        # combine train+val samples from subsets A and B
        train_and_val = subset_a + b_train_val
        random.shuffle(train_and_val) # shuffle the combined pool

        train_samples = int(len(train_and_val) * 0.75) # train is 75% of train and val
        val_samples = len(train_and_val) - train_samples # take the rest for validation (25% of train and val)
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])
        # save information about train dataset
        train.dimensions = self.properties_dim

        print("\nFinal Dataset Sizes:")
        print(f"Train dataset size: {len(train)}")
        print(f"Validation dataset size: {len(val)}")
        print(f"Test dataset size: {len(test)}")

        return train, val, test

    def get_item(self, concept_idx, context_condition, encoding_func, include_concept=False):
        """
        Receives concept-context pairs and an encoding function.
        Returns encoded (sender_input, labels, receiver_input).
            sender_input: (sender_input_objects, sender_labels)
            labels: indices of target objects in the receiver_input
            receiver_input: receiver_input_objects
        The sender_input_objects and the receiver_input_objects are different objects sampled from the same concept
        and context condition.
        """
        if self.shared_context:
            (sender_concept, sender_context,
             receiver_concept, receiver_context) = self.get_shared_context(concept_idx, context_condition)
        else:
            # use get_sample() to get sampled target and distractor objects
            # The concrete sampled objects can differ between sender and receiver.
            sender_concept, sender_context = self.get_sample(concept_idx, context_condition)
            receiver_concept, receiver_context = self.get_sample(concept_idx, context_condition)
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
        receiver_label = [idx for idx, obj in enumerate(receiver_input) if obj in receiver_targets]
        receiver_label = torch.Tensor(receiver_label).to(torch.int64).to(device=self.device)
        receiver_label = F.one_hot(receiver_label, num_classes=self.game_size * 2).sum(dim=0).float()
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
        To be used separately for sender and receiver.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        # sample target objects for given game size (if possible, get unique choices)
        try:
            target_objects = random.sample(all_target_objects, self.game_size)
        except ValueError:
            target_objects = random.choices(all_target_objects, k=self.game_size)
        # get all possible distractors for a given concept (for all possible context conditions)
        context = self.get_distractors(concept_idx, context_condition)
        context_sampled = self.sample_distractors(context, context_condition)
        # return target concept, context (distractor objects + context_condition) for each context
        return [target_objects, fixed], context_sampled

    def get_shared_context(self, concept_idx, context_condition):
        """
        Returns a full sample consisting of a set of target objects (target concept)
        and a set of distractor objects (context) for a given concept condition.
        This can be used to create a shared context between sender and receiver.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        for interlocutor in ['sender', 'receiver']:
            # sample target objects for given game size (if possible, get unique choices)
            try:
                target_objects = random.sample(all_target_objects, self.game_size)
            except ValueError:
                target_objects = random.choices(all_target_objects, k=self.game_size)
            # get all possible distractors for a given concept (for all possible context conditions)
            if interlocutor == 'sender':
                context, shared_attr_indices = self.get_distractors_shared(concept_idx, context_condition)
            else:
                context, _ = self.get_distractors_shared(concept_idx, context_condition, shared_attr_indices)

            context_sampled = self.sample_distractors(context, context_condition)

            if interlocutor == 'sender':
                s_target_objects = target_objects
                s_context_sampled = context_sampled
            else:
                r_target_objects = target_objects
                r_context_sampled = context_sampled

        # return target concept, context (distractor objects + context_condition) for each context
        return [s_target_objects, fixed], s_context_sampled, [r_target_objects, fixed], r_context_sampled

    def get_distractors(self, concept_idx, context_condition):
        """
        Computes distractors.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        context = []

        # save fixed attribute indices in a list for later comparisons
        fixed_attr_indices = []
        for index, value in enumerate(fixed):
            if value == 1:
                fixed_attr_indices.append(index)

        # consider all objects as possible distractors
        poss_dist = self.all_objects

        for obj in poss_dist:
            # find out how many attributes are shared between the possible distractor object and the target concept
            # (by only comparing fixed attributes because only these are relevant for defining the context)
            shared = sum(1 for idx in fixed_attr_indices if obj[idx] == all_target_objects[0][idx])
            if shared == context_condition:
                context.append(obj)

        return context

    def get_distractors_shared(self, concept_idx, context_condition, shared_attr_indices=None, split_by_attribute_subset_a=False):
        """
        Computes distractors for a shared context between sender and receiver. It also implements a change compared
        to get_distractors(): Here, objects in the context do not only share a certain number of attributes, but also
        the position of the attributes that are shared are fixed (and shared between sender and receiver).
        split_by_attribute_subset_a: used for split_by_attribute to manipulate which attributes can be sahred between targets
        and distractors for subset A.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        context = []

        # save fixed attribute indices in a list for later comparisons
        fixed_attr_indices = []
        for index, value in enumerate(fixed):
            if value == 1:
                fixed_attr_indices.append(index)

        # if not given, compute (this is the case for the sender)
        if not shared_attr_indices:
            """
            Generate a shared vector, i.e. a vector that indicates which of the attributes should be shared with the
            concept attributes according to the context condition
            e.g. if fixed==(1,1,1) and context_condition==2, then shared vector can be one of: (1,1,0), (1,0,1), (0,1,1)
            If the dataset is generated with split_by_attribute: for concepts-context pairs in subset A,
            the discriminative attribute cannot be shared between targets and distractors.
            Meaning, if the discriminative attribute is fixed and the context granularity is "fine",
            the concept-context pair should go into subset A, and discriminative attribute cannot be sahred with distractors.
            """
            if split_by_attribute_subset_a and self.discriminative_attribute in fixed_attr_indices and context_condition == sum(fixed) -1:
                 # exclude the discriminative attribute from the pool of attributes to be shared
                 attributes_to_be_shared = [attr for attr in fixed_attr_indices if attr != self.discriminative_attribute]
                 shared_attr_indices = random.sample(attributes_to_be_shared, context_condition)
            else:
                 shared_attr_indices = random.sample(fixed_attr_indices, context_condition)

        # consider all objects as possible distractors
        poss_dist = self.all_objects

        for obj in poss_dist:
            # find out how many attributes are shared between the possible distractor object and the target concept
            # (by only comparing fixed attributes because only these are relevant for defining the context)
            shared = sum(1 for idx in fixed_attr_indices if obj[idx] == all_target_objects[0][idx])
            # this still works for context condition 0:
            if len(shared_attr_indices) == 0:
                if shared == context_condition:
                    context.append(obj)
            else:
                if shared == context_condition:  # this filters out already some of the objects to keep the loop as small as possible
                    if np.all(
                            np.array(obj)[shared_attr_indices] == np.array(all_target_objects[0])[shared_attr_indices]):
                        context.append(obj)

        return context, shared_attr_indices

    def sample_distractors(self, context, context_condition):
        """
        Function for sampling the distractors from a specified context condition.
        """
        # sample distractor objects for given game size and the specified context condition
        # distractors = [dist_obj for dist_objs in context for dist_obj in dist_objs]
        context_new = []
        try:
            context_new.append([random.sample(context, self.game_size), context_condition])
        except ValueError:
            context_new.append([random.choices(context, k=self.game_size), context_condition])
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
                        context.append([random.sample(helper_list, self.game_size), helper_i])
                    except ValueError:
                        context.append([random.choices(helper_list, k=self.game_size), helper_i])
            # catch the final context condition as well
            elif context_condition == len(self.properties_dim) - 1:
                try:
                    context.append([random.sample(helper_list, self.game_size), helper_i])
                except ValueError:
                    context.append([random.choices(helper_list, k=self.game_size), helper_i])
                helper_i = helper_i + 1
                helper_list = list()
                helper_list.append(dist_object)
            # when moving to the next context condition, first sample from the old
            else:
                # sample from all objects belonging to the same context condition
                try:
                    context.append([random.sample(helper_list, self.game_size), helper_i])
                except ValueError:
                    context.append([random.choices(helper_list, k=self.game_size), helper_i])
                helper_i = helper_i + 1
                helper_list = list()
                helper_list.append(dist_object)
        return context

    def get_all_concepts(self):
        """
        Returns all possible concepts for a given dataset size.
        Concepts consist of (objects, fixed) tuples
            objects: a list with all object-tuples that satisfy the concept
            fixed: a tuple that denotes how many and which attributes are fixed
        """
        fixed_vectors = self.get_fixed_vectors(self.properties_dim)
        # create all possible concepts
        all_fixed_object_pairs = list(itertools.product(self.all_objects, fixed_vectors))

        concepts = list()
        # go through all concepts (i.e. fixed, objects pairs)
        for concept in all_fixed_object_pairs:
            # treat each fixed_object pair as a target concept once
            # e.g. target concept (_, _, 0) (i.e. fixed = (0,0,1) and objects e.g. (0,0,0), (1,0,0))
            fixed = concept[1]
            # go through all objects and check whether they satisfy the target concept (in this example have 0 as 3rd attribute)
            target_objects = list()
            for object in self.all_objects:
                if self.satisfies(object, concept):
                    if object not in target_objects:
                        target_objects.append(object)
            # concepts are tuples of fixed attributes and all target objects that satisfy the concept
            if (target_objects, fixed) not in concepts:
                concepts.append((target_objects, fixed))
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

    def reverse_one_hot(self):

        # dictionary to translate one-hot encoded labels for shapes3d dataset
        # only used for shapes3d dataset variant
        attribute_dict = {
            0: (0.0, 0.75, 0.0),
            1: (0.0, 0.75, 1.0),
            2: (0.0, 0.75, 2.0),
            3: (0.0, 0.75, 3.0),
            4: (0.0, 0.9642857142857143, 0.0),
            5: (0.0, 0.9642857142857143, 1.0),
            6: (0.0, 0.9642857142857143, 2.0),
            7: (0.0, 0.9642857142857143, 3.0),
            8: (0.0, 1.1071428571428572, 0.0),
            9: (0.0, 1.1071428571428572, 1.0),
            10: (0.0, 1.1071428571428572, 2.0),
            11: (0.0, 1.1071428571428572, 3.0),
            12: (0.0, 1.25, 0.0),
            13: (0.0, 1.25, 1.0),
            14: (0.0, 1.25, 2.0),
            15: (0.0, 1.25, 3.0),
            16: (0.2, 0.75, 0.0),
            17: (0.2, 0.75, 1.0),
            18: (0.2, 0.75, 2.0),
            19: (0.2, 0.75, 3.0),
            20: (0.2, 0.9642857142857143, 0.0),
            21: (0.2, 0.9642857142857143, 1.0),
            22: (0.2, 0.9642857142857143, 2.0),
            23: (0.2, 0.9642857142857143, 3.0),
            24: (0.2, 1.1071428571428572, 0.0),
            25: (0.2, 1.1071428571428572, 1.0),
            26: (0.2, 1.1071428571428572, 2.0),
            27: (0.2, 1.1071428571428572, 3.0),
            28: (0.2, 1.25, 0.0),
            29: (0.2, 1.25, 1.0),
            30: (0.2, 1.25, 2.0),
            31: (0.2, 1.25, 3.0),
            32: (0.4, 0.75, 0.0),
            33: (0.4, 0.75, 1.0),
            34: (0.4, 0.75, 2.0),
            35: (0.4, 0.75, 3.0),
            36: (0.4, 0.9642857142857143, 0.0),
            37: (0.4, 0.9642857142857143, 1.0),
            38: (0.4, 0.9642857142857143, 2.0),
            39: (0.4, 0.9642857142857143, 3.0),
            40: (0.4, 1.1071428571428572, 0.0),
            41: (0.4, 1.1071428571428572, 1.0),
            42: (0.4, 1.1071428571428572, 2.0),
            43: (0.4, 1.1071428571428572, 3.0),
            44: (0.4, 1.25, 0.0),
            45: (0.4, 1.25, 1.0),
            46: (0.4, 1.25, 2.0),
            47: (0.4, 1.25, 3.0),
            48: (0.8, 0.75, 0.0),
            49: (0.8, 0.75, 1.0),
            50: (0.8, 0.75, 2.0),
            51: (0.8, 0.75, 3.0),
            52: (0.8, 0.9642857142857143, 0.0),
            53: (0.8, 0.9642857142857143, 1.0),
            54: (0.8, 0.9642857142857143, 2.0),
            55: (0.8, 0.9642857142857143, 3.0),
            56: (0.8, 1.1071428571428572, 0.0),
            57: (0.8, 1.1071428571428572, 1.0),
            58: (0.8, 1.1071428571428572, 2.0),
            59: (0.8, 1.1071428571428572, 3.0),
            60: (0.8, 1.25, 0.0),
            61: (0.8, 1.25, 1.0),
            62: (0.8, 1.25, 2.0),
            63: (0.8, 1.25, 3.0)}

        indeces = np.argmax(self.labels, axis=1)
        unhottified = []

        for index in indeces:
            unhottified.append(attribute_dict[index])

        return unhottified

    @staticmethod
    def satisfies(object, concept):
        """
        Checks whether an object satisfies a target concept, returns a boolean value.
        Concept consists of an object vector and a fixed vector tuple.
        """
        satisfied = False
        same_counter = 0
        concept_object, fixed = concept
        # an object satisfies a concept if fixed attributes are the same
        # go through attributes an check whether they are fixed
        for i, attr in enumerate(fixed):
            # if an attribute is fixed
            if attr == 1:
                # compare object with concept object
                if object[i] == concept_object[i]:
                    same_counter = same_counter + 1
        # the number of shared attributes should match the number of fixed attributes
        if same_counter == sum(fixed):
            satisfied = True
        return satisfied

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
    def get_all_objects_for_a_concept(properties_dim, features, fixed):
        """
        Returns all possible objects for a concept at a given level of abstraction
        features: Defines the features which are fixed
        fixed: Defines how many and which attributes are fixed
        """
        # retrieve all possible objects
        list_of_dim = [range(0, dim) for dim in properties_dim]
        all_objects = list(itertools.product(*list_of_dim))

        # get concept objects
        concept_objects = list()

        # account for the case where 0 attributes should be shared in context_condition 0
        if not 1 in fixed:
            return all_objects

        # determine the indices of attributes that should be fixed
        fixed_indices = list(itertools.compress(range(0, len(fixed)), fixed))
        # find possible concepts for each index
        possible_concepts = dict()
        for index in fixed_indices:
            possible_concepts[index] = ([object for object in all_objects if object[index] == features[index]])

        # keep only those that also match with the other fixed features, i.e. that are possible concepts for all fixed indices
        all = list(possible_concepts.values())
        concept_objects = list(set(all[0]).intersection(*all[1:]))

        return concept_objects

    @staticmethod
    def _get_all_possible_objects(properties_dim):
        """
        Returns all possible combinations of attribute-feature values as a dataframe.
        """
        list_of_dim = [range(0, dim) for dim in properties_dim]
        # Each object is a row
        all_objects = list(itertools.product(*list_of_dim))
        return all_objects  # pd.DataFrame(all_objects)

    def _many_hot_encoding(self, input_list):
        """
        Outputs a binary one dim vector
        """
        output = torch.zeros([sum(self.properties_dim)]).to(device=self.device)
        start = 0

        for elem, dim in zip(input_list, self.properties_dim):
            output[start + elem] = 1
            start += dim

        return output

    def _sample_image_from_concept(self, concept):
        all_objects = self.reverse_one_hot()
        indices = np.where(np.all(all_objects == np.array(concept), axis=1))[0].tolist()
        random_index = random.choice(indices) if indices else None
        sampled_img = self.images[random_index] if random_index is not None else None
        return torch.tensor(sampled_img, dtype=torch.float32, device=self.device)


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
        while (n_attributes > 0):
            # if changed_concepts is empty, I consider the target_object
            if not changed_concepts:
                changed_concepts = [change_one_attribute(input_object, fixed)]
                n_attributes = n_attributes - 1
            # otherwise consider the changed concepts and change them again	 until n_attributes = 0
            else:
                old_changed_concepts = changed_concepts.copy()
                # O(game_size)
                for sublist in changed_concepts:
                    for (changed_concept, fixed) in sublist:
                        new_changed_concepts = change_one_attribute(changed_concept, fixed)
                        if new_changed_concepts not in old_changed_concepts:
                            old_changed_concepts.append(new_changed_concepts)
                # copy and store for next iteration
                changed_concepts = old_changed_concepts.copy()
                n_attributes = n_attributes - 1
        # flatten list
        changed_concepts_flattened = [changed_concept for sublist in changed_concepts for changed_concept in sublist]
        # remove doubles
        changed_concepts_final = []
        [changed_concepts_final.append(x) for x in changed_concepts_flattened if x not in changed_concepts_final]
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
            [(self.get_all_objects_for_a_concept(self.properties_dim, dist_concept[0], fixed), tuple(dist_concept[1]))])

    return distractor_objects
