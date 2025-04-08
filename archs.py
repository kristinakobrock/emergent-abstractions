import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# try out different senders:
# 1) a sender who only sees the targets
# 2) a sender who receives the objects in random order and a vector of labels indicating which are the targets
# 3) a sender who computes prototype embeddings over targets and distractors
# 4) a sender who receives targets first, then distractors and is thus implicitly informed 
# about which are the targets (used in Lazaridou et al. 2017)


class Sender(nn.Module):
    """
    Sender gets as input targets and distractors in an ordered fashion (targets first).
    It embeds both targets and distractors and returns a joint embedding.
    """

    def __init__(self, n_hidden, n_features, n_targets, context_unaware=False, shapes3d=False):
        super(Sender, self).__init__()
        self.context_unaware = context_unaware
        self.shapes3d = shapes3d
        # embedding layers:
        self.fc1 = nn.Linear(n_features * n_targets, n_hidden)
        self.fc2 = nn.Linear(n_features * n_targets, n_hidden)
        self.fc3 = nn.Linear(2 * n_hidden, n_hidden)
        if self.shapes3d:
            self.bn1 = nn.BatchNorm1d(n_hidden)
            self.dropout = nn.Dropout(0.2)

    def forward(self, x, aux_input=None):
        batch_size = x.shape[0]
        n_obj = x.shape[1]
        n_features = x.shape[2]
        n_targets = int(n_obj / 2)

        # Ensure the input tensor has the correct shape
        assert n_features == 100, f"Expected n_features to be 100, but got {n_features}"
        #assert n_targets == 10, f"Expected n_targets to be 10, but got {n_targets}"

        # embed target objects:
        targets = x[:, :n_targets]
        targets_flat = targets.reshape(batch_size, n_targets * n_features)
        target_feature_embedding = F.relu(self.fc1(targets_flat))

        # context unaware speakers only process the targets
        if self.context_unaware:
            return target_feature_embedding

        # context aware speakers process both targets and distractors
        else:
            # embed distractor objects:
            distractors = x[:, n_targets:]
            distractors_flat = distractors.reshape(batch_size, n_targets * n_features)
            distractor_feature_embedding = F.relu(self.fc2(distractors_flat))

            # create joint embedding
            if self.shapes3d:
                joint_embedding = self.fc3(torch.cat([target_feature_embedding, distractor_feature_embedding], dim=1))
                joint_embedding = self.bn1(joint_embedding).tanh() # [32, 1024]
                joint_embedding = self.dropout(joint_embedding)
            else:
                joint_embedding = self.fc3(torch.cat([target_feature_embedding, distractor_feature_embedding], dim=1)).tanh()
            return joint_embedding


class Receiver(nn.Module):
    def __init__(self, n_features, n_hidden, shapes3d=False):
        """
        Receives as input the vector generated by the message-decoding RNN in the wrapper (x)
        and game-specific input (input, i.e. matrix containing all input attribute-value vectors).
        The module maps these vectors to the same dimensionality as the RNN output vector, 
        and computes a dot product between the latter and each of the (transformed) input vectors.
        The output dot product list is interpreted as a non-normalized probability distribution 
        over possible positions of the target.
        """
        super(Receiver, self).__init__()
        self.shapes3d = shapes3d
        # embedding layer
        self.fc1 = nn.Linear(n_features, n_hidden)
        if self.shapes3d:
            self.bn1 = nn.BatchNorm1d(n_hidden)
            self.dropout = nn.Dropout(0.2)

    def forward(self, x, input, _aux_input=None):
        # from EGG: the rationale for the non-linearity here is that the RNN output will also be the 
        # outcome of a non-linearity
        embedded_input = self.fc1(input)
        if self.shapes3d:
            embedded_input = self.dropout(embedded_input)
            embedded_input = self.bn1(embedded_input.swapaxes(1, 2)).swapaxes(1, 2)
        embedded_input = embedded_input.tanh()  # [32, 20, 256]
        dots = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return dots.squeeze()  # [32, 20]


class RSASender(nn.Module):
    """
    The RSA sender uses an internal listener to calculate the utility of each utterance and selects the utterance
    with the highest utility. The utility is calculated as the difference between the score of the targets and the
    score of the distractors minus the cost of the utterance (message length).
    Score = sum of unnormalized probabilities
    Cost = position of the first EOS symbol in the message.

    As a sender, it gets as input targets and distractors in an ordered fashion (targets first).
    """

    def __init__(self, internal_listener, utterances, cost_factor=0.01):
        """
        internal_listener: Trained listener. Needs to be able to process one-hot encoded messages
        utterances: List of possible utterances to pick from. Each utterance is a one-hot encoded tensor
        cost_factor: Weighting factor for the cost of the utterance (message length)
        """
        super(RSASender, self).__init__()
        self.internal_listener = internal_listener
        self.utterances = utterances
        self.cost_factor = cost_factor

    def forward(self, x, aux_input=None):
        batch_size = x.shape[0]
        num_utterances = len(self.utterances)
        utilities = torch.zeros(batch_size, num_utterances)

        for i, utterance in enumerate(self.utterances):
            utility = self.calculate_utility(x, utterance)
            utilities[:, i] = utility

        # Find the indices of utterances with the highest utility for each item in the batch
        max_indices = torch.argmax(utilities, dim=1)

        # Collect the best utterance for each batch item
        best_utterances = [self.utterances[i] for i in max_indices]
        return torch.stack(best_utterances)

    def calculate_utility(self, x, utterance):
        """
        Calculate the utility of an utterance given the batch of inputs x.
        The utterance is repeated to match the batch size and the utility is calculated as the difference between
        the score of the targets and the score of the distractors minus the cost of the utterance.
        """
        batch_size = x.shape[0]

        # Find effective message length (first occurrence of EOS symbol)
        eos_one_hot = torch.zeros_like(utterance[0]).float()
        eos_one_hot[0] = 1
        matches = torch.all(utterance == eos_one_hot, dim=1)
        zeroes = torch.nonzero(matches)
        cost = zeroes[0] if zeroes.shape[0] > 0 else len(utterance[0])

        # Repeat the utterance to match the batch size. All inputs in batch will be paired with the same utterance.
        utterance = utterance.unsqueeze(0).repeat(batch_size, 1, 1)
        logits = self.internal_listener(input=x, message=utterance)

        # Get number of targets so we can slice the logits into targets and distractors
        n_obj = x.shape[1]
        n_targets = int(n_obj / 2)
        # Using -1 to get the logits for the last RNN time step
        # KK: added "/n_target" for averaging instead of summing. With summing, the values get very large (-150) which
        # is not good with respect to the cost calculation (which only subtracts a very small value in comparison).
        target_score = torch.sum(logits[:, -1, :n_targets], dim=1)/n_targets
        distractor_score = torch.sum(logits[:, -1, n_targets:], dim=1)/n_targets
        overall_score = target_score - distractor_score

        return overall_score - self.cost_factor * cost


class RSAReceiver(nn.Module):
    """
    The RSA receiver uses an internal speaker model which is an RSASender, i.e. a pragmatic speaker that produces
    utterances based on their calculated utility. The pragmatic speaker has to be context-aware.

    As a receiver, it gets as input a message from a speaker as well as targets and distractors in a random order. It
    has to predict the correct labels for the input objects (i.e. target/distractor).
    """

    def __init__(self, internal_speaker):
        """
        internal_speaker: Trained speaker. Needs to be able to process one-hot encoded messages
        """
        super(RSAReceiver, self).__init__()
        self.internal_speaker = internal_speaker

    def forward(self, x, input, aux_input=None):
        batch_size = x.shape[0]
        n_obj = x.shape[1]
        n_targets = int(n_obj / 2)
        for i in range(batch_size):
            for j in range(n_obj):
        #    utility = self.calculate_utility(x, utterance)
        # Repeat the utterance to match the batch size. All inputs in batch will be paired with the same utterance.
                speaker_input = x[i][j].unsqueeze(0).repeat(batch_size, 1, 1)
                message = self.internal_speaker(speaker_input)

        # The pragmatic listener reasons about a pragmatic speaker.
        # The pragmatic listener receives an utterance (input) and a set of objects (x).
        # The prediction is made by inputting the utterance into the internal pragmatic speaker model and choosing the
        # objects as targets accordingly.

        # Now, it can happen that the objects output by the speaker are not exactly the same as the objects in x
        # So, we need to compute some similarity score and output the most probable predictions based on that.

        # The pragmatic listener reasons about a pragmatic speaker, i.e. wants to know for each conceivable state of the
        # world the pragmatic speaker would choose which utterance
        # pretend that state 1 is the state of the world and ask the pragmatic speaker what they would call it
        # configure each possible state of the world
        # the following assumes that the agents know that there are exactly 10 out of 20 targets
        # predictions = list(itertools.combinations_with_replacement([0, 1], n_obj))
        # # combine these predictions with the actual objects, i.e. put the objects in a target-first order that
        # # corresponds to the labels
        # for i, prediction in enumerate(predictions):
        #     indices = np.where(prediction == 1)[0]
        #     sorted = x[np.argsort(indices)]
        #     best_utterance = self.pragmatic_speaker(sorted)
        #     # compare the utterance to the actually produced utterance (input)
        #     # using some similarity score? And if it exceeds a certain value, we settle for the prediction?

        # How should the prediction look like? -1 and +1 ?
        return input

