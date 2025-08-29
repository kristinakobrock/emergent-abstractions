import os.path
import pickle
import numpy as np
import copy


def load_accuracies(all_paths, n_runs=5, n_epochs=300, val_steps=10, zero_shot=False, context_unaware=False,
                    length_cost=False, early_stopping=False, rsa=False, rsa_test=None, zero_shot_test_ds=None,
                    sampled_context=False, hierarchical=False, shared_context=False, split_by_attribute=False, split_test_ds=None):
    """ loads all accuracies into a dictionary, val_steps should be set to the same as val_frequency during training
    """
    result_dict = {'train_acc': [], 'val_acc': [], 'test_acc': [],
                   'train_message_lengths': [], 'val_message_lengths': [],
                   'zs_specific_train_acc': [], 'zs_specific_val_acc': [], 'zs_specific_test_acc': [],
                   'zs_specific_train_message_length': [], 'zs_specific_val_message_length': [],
                   'zs_generic_train_acc': [], 'zs_generic_val_acc': [], 'zs_generic_test_acc': [],
                   'zs_generic_train_message_length': [], 'zs_generic_val_message_length': [],
                   'cu_train_acc': [], 'cu_val_acc': [], 'cu_test_acc': [],
                   'cu_train_message_lengths': [], 'cu_val_message_lengths': [],
                   'cu_zs_specific_train_acc': [], 'cu_zs_specific_val_acc': [], 'cu_zs_specific_test_acc': [],
                   'cu_zs_specific_val_message_length': [], 'cu_zs_generic_train_message_length': [],
                   'cu_zs_generic_train_acc': [], 'cu_zs_generic_val_acc': [], 'cu_zs_generic_test_acc': [],
                   'cu_zs_specific_train_message_length': [], 'cu_zs_generic_val_message_length': [],
                   'rsa_test_loss': [], 'rsa_test_acc': [], 'final_test_acc': [], 'final_test_loss': [],
                   'rsa_test_gen_utt_loss': [], 'rsa_test_gen_utt_acc': [],
                   'zs_specific_rsa_test_loss': [], 'zs_specific_rsa_test_acc': [],
                   'zs_specific_final_test_acc': [], 'zs_specific_final_test_loss': [],
                   'zs_generic_rsa_test_loss': [], 'zs_generic_rsa_test_acc': [],
                   'zs_generic_final_test_acc': [], 'zs_generic_final_test_loss': []}

    for path_idx, path in enumerate(all_paths):

        train_accs = []
        val_accs = []
        test_accs = []
        train_message_lengths = []
        val_message_lengths = []
        zs_specific_train_accs = []
        zs_specific_val_accs = []
        zs_specific_test_accs = []
        zs_specific_train_message_lengths = []
        zs_specific_val_message_lengths = []
        zs_generic_train_accs = []
        zs_generic_val_accs = []
        zs_generic_test_accs = []
        zs_generic_train_message_lengths = []
        zs_generic_val_message_lengths = []
        cu_train_accs = []
        cu_val_accs = []
        cu_test_accs = []
        cu_train_message_lengths = []
        cu_val_message_lengths = []
        cu_zs_specific_train_accs = []
        cu_zs_specific_val_accs = []
        cu_zs_specific_test_accs = []
        cu_zs_specific_train_message_lengths = []
        cu_zs_specific_val_message_lengths = []
        cu_zs_generic_train_accs = []
        cu_zs_generic_val_accs = []
        cu_zs_generic_test_accs = []
        cu_zs_generic_train_message_lengths = []
        cu_zs_generic_val_message_lengths = []
        rsa_test_losses = []
        rsa_test_accs = []
        final_test_accs = []
        final_test_losses = []
        rsa_test_gen_utt_losses = []
        rsa_test_gen_utt_accs = []
        zs_specific_rsa_test_losses = []
        zs_specific_rsa_test_accs = []
        zs_specific_final_test_accs = []
        zs_specific_final_test_losses = []
        zs_generic_rsa_test_losses = []
        zs_generic_rsa_test_accs = []
        zs_generic_final_test_accs = []
        zs_generic_final_test_losses = []


        # prepare paths
        if sampled_context:
            standard_path = "standard/sampled_context"
            context_unaware_path = "context_unaware/sampled_context"
        else:
            standard_path = "standard"
            context_unaware_path = "context_unaware"
        context_aware_path = "context_aware"
        if hierarchical:
            context_unaware_path = context_unaware_path + '/hierarchical'
            context_aware_path = context_aware_path + '/hierarchical'
        if shared_context and not split_by_attribute:
            context_unaware_path = context_unaware_path + '/shared_context'
            context_aware_path = context_aware_path + '/shared_context'
        if split_by_attribute:
            context_unaware_path = context_unaware_path + '/split_by_attribute'
            context_aware_path = context_aware_path + '/split_by_attribute'
        length_cost_path = "length_cost"
        zero_shot_path = "zero_shot"
        if rsa:
            rsa_file_extension = rsa_test
        file_name = "loss_and_metrics"
        file_name_zs_default = "loss_and_metrics"
        file_name_split = ""
        if zero_shot_test_ds is not None:
            file_name_zs = str("loss_and_metrics_" + zero_shot_test_ds)
        if split_test_ds is not None:
            file_name_split = str("loss_and_metrics_" + split_test_ds)
        file_extension = "pkl"

        for run in range(n_runs):

            run_path = str(run)

            # context-aware (standard)
            if not context_unaware and not length_cost and not zero_shot:
                if rsa_test is None:
                    file_path = f"{path}/{run_path}/{file_name}.{file_extension}"
                    data = pickle.load(open(file_path, 'rb'))
                    # train and validation accuracy
                    lists = sorted(data['metrics_train0'].items())
                    _, train_acc = zip(*lists)
                    train_accs.append(train_acc)
                    lists = sorted(data['metrics_test0'].items())
                    _, val_acc = zip(*lists)
                    if (len(val_acc) > n_epochs // val_steps) and not early_stopping:  # old: we had some runs where we set val freq to 5 instead of 10
                        val_acc = val_acc[::2]
                    val_accs.append(val_acc)
                    if split_test_ds is not None:
                        file_path_test = f"{path}/{run_path}/{file_name_split}.{file_extension}"
                        data_test = pickle.load(open(file_path_test, 'rb'))
                        test_accs.append(data_test['final_test_acc'])
                    # message lengths
                    lists = sorted(data['metrics_train1'].items())
                    _, train_message_length = zip(*lists)
                    lists = sorted(data['metrics_test1'].items())
                    _, val_message_length = zip(*lists)
                    train_message_lengths.append(train_message_length)
                    val_message_lengths.append(val_message_length)
                else:
                    file_path = f"{path}/{standard_path}/{run_path}/{file_name}_{rsa_file_extension}.{file_extension}"
                    data = pickle.load(open(file_path, 'rb'))
                    # test acc
                    final_test_accs.append(data['final_test_acc'])
                    final_test_losses.append(data['final_test_loss'])
                    rsa_test_accs.append(data['rsa_test_acc'])
                    rsa_test_losses.append(data['rsa_test_loss'])
                    try:
                        rsa_test_gen_utt_accs.append(data['rsa_test_gen_utt_acc'])
                        rsa_test_gen_utt_losses.append(data['rsa_test_gen_utt_loss'])
                    except KeyError:
                        pass

            # context-unaware
            elif context_unaware and not length_cost and not zero_shot:
                if rsa_test is None:
                    file_path = f"{path}/{context_unaware_path}/{run_path}/{file_name}.{file_extension}"
                    cu_data = pickle.load(open(file_path, 'rb'))
                    # accuracies
                    lists = sorted(cu_data['metrics_train0'].items())
                    _, cu_train_acc = zip(*lists)
                    if (len(cu_train_acc) != n_epochs) and not early_stopping:
                        print(path, run, len(cu_train_acc))
                        raise ValueError(
                            "The stored results don't match the parameters given to this function. "
                            "Check the number of epochs in the above mentioned runs.")
                    cu_train_accs.append(cu_train_acc)
                    lists = sorted(cu_data['metrics_test0'].items())
                    _, cu_val_acc = zip(*lists)
                    # for troubleshooting in case the stored results don't match the parameters given to this function
                    if (len(cu_val_acc) != n_epochs // val_steps) and not early_stopping:
                        print(context_unaware_path, len(cu_val_acc))
                        raise ValueError(
                            "The stored results don't match the parameters given to this function. "
                            "Check the above mentioned files for number of epochs and validation steps.")
                    if (len(cu_val_acc) > n_epochs // val_steps) and not early_stopping:
                        cu_val_acc = cu_val_acc[::2]
                    cu_val_accs.append(cu_val_acc)
                    cu_test_accs.append(cu_data['final_test_acc'])
                    # message lengths
                    lists = sorted(cu_data['metrics_train1'].items())
                    _, cu_train_message_length = zip(*lists)
                    lists = sorted(cu_data['metrics_test1'].items())
                    _, cu_val_message_length = zip(*lists)
                    cu_train_message_lengths.append(cu_train_message_length)
                    cu_val_message_lengths.append(cu_val_message_length)
                else:
                    file_path = f"{path}/{context_unaware_path}/{run_path}/{file_name}_{rsa_file_extension}.{file_extension}"
                    data = pickle.load(open(file_path, 'rb'))
                    # test acc
                    final_test_accs.append(data['final_test_acc'])
                    final_test_losses.append(data['final_test_loss'])
                    rsa_test_accs.append(data['rsa_test_acc'])
                    rsa_test_losses.append(data['rsa_test_loss'])
                    try:
                        rsa_test_gen_utt_accs.append(data['rsa_test_gen_utt_acc'])
                        rsa_test_gen_utt_losses.append(data['rsa_test_gen_utt_loss'])
                    except KeyError:
                        pass

            # length cost
            elif length_cost and not zero_shot:
                if not context_unaware:
                    if rsa_test is None:
                        file_path = f"{path}/{length_cost_path}/{context_aware_path}/{run_path}/{file_name}.{file_extension}"
                        # train and validation accuracy
                        data = pickle.load(open(file_path, 'rb'))
                        lists = sorted(data['metrics_train0'].items())
                        _, train_acc = zip(*lists)
                        train_accs.append(train_acc)
                        lists = sorted(data['metrics_test0'].items())
                        _, val_acc = zip(*lists)
                        if len(val_acc) > n_epochs // val_steps and not early_stopping:  # old: we had some runs where we set val freq to 5 instead of 10
                            val_acc = val_acc[::2]
                        val_accs.append(val_acc)
                        test_accs.append(data['final_test_acc'])
                        # message lengths
                        lists = sorted(data['metrics_train1'].items())
                        _, train_message_length = zip(*lists)
                        lists = sorted(data['metrics_test1'].items())
                        _, val_message_length = zip(*lists)
                        train_message_lengths.append(train_message_length)
                        val_message_lengths.append(val_message_length)
                    else:
                        file_path = f"{path}/{length_cost_path}/{context_aware_path}/{run_path}/{file_name}_{rsa_file_extension}.{file_extension}"
                        data = pickle.load(open(file_path, 'rb'))
                        # test acc
                        final_test_accs.append(data['final_test_acc'])
                        final_test_losses.append(data['final_test_loss'])
                        rsa_test_accs.append(data['rsa_test_acc'])
                        rsa_test_losses.append(data['rsa_test_loss'])
                        try:
                            rsa_test_gen_utt_accs.append(data['rsa_test_gen_utt_acc'])
                            rsa_test_gen_utt_losses.append(data['rsa_test_gen_utt_loss'])
                        except KeyError:
                            pass

                else:

                    if rsa_test is None:
                        file_path = f"{path}/{length_cost_path}/{context_unaware_path}/{run_path}/{file_name}.{file_extension}"
                        cu_data = pickle.load(open(file_path, 'rb'))
                        # accuracies
                        lists = sorted(cu_data['metrics_train0'].items())
                        _, cu_train_acc = zip(*lists)
                        if (len(cu_train_acc) != n_epochs) and not early_stopping:
                            print(path, run, len(cu_train_acc))
                            raise ValueError(
                                "The stored results don't match the parameters given to this function. "
                                "Check the number of epochs in the above mentioned runs.")
                        cu_train_accs.append(np.array(cu_train_acc))
                        lists = sorted(cu_data['metrics_test0'].items())
                        _, cu_val_acc = zip(*lists)
                        # for troubleshooting in case the stored results don't match the parameters given to this function
                        if (len(cu_val_acc) != n_epochs // val_steps) and not early_stopping:
                            print(context_unaware_path, len(cu_val_acc))
                            raise ValueError(
                                "The stored results don't match the parameters given to this function. "
                                "Check the above mentioned files for number of epochs and validation steps.")
                        if (len(cu_val_acc) > n_epochs // val_steps) and not early_stopping:
                            cu_val_acc = cu_val_acc[::2]
                        cu_val_accs.append(cu_val_acc)
                        cu_test_accs.append(cu_data['final_test_acc'])
                        # message lengths
                        lists = sorted(cu_data['metrics_train1'].items())
                        _, cu_train_message_length = zip(*lists)
                        lists = sorted(cu_data['metrics_test1'].items())
                        _, cu_val_message_length = zip(*lists)
                        cu_train_message_lengths.append(cu_train_message_length)
                        cu_val_message_lengths.append(cu_val_message_length)
                    else:
                        file_path = f"{path}/{length_cost_path}/{context_unaware_path}/{run_path}/{file_name}_{rsa_file_extension}.{file_extension}"
                        data = pickle.load(open(file_path, 'rb'))
                        # test acc
                        final_test_accs.append(data['final_test_acc'])
                        final_test_losses.append(data['final_test_loss'])
                        rsa_test_accs.append(data['rsa_test_acc'])
                        rsa_test_losses.append(data['rsa_test_loss'])
                        try:
                            rsa_test_gen_utt_accs.append(data['rsa_test_gen_utt_acc'])
                            rsa_test_gen_utt_losses.append(data['rsa_test_gen_utt_loss'])
                        except KeyError:
                            pass

            # zero_shot
            elif zero_shot:
                for cond in ['specific', 'generic']:
                    if not context_unaware:
                        if not length_cost:
                            if zero_shot_test_ds is None:
                                file_path = f"{path}/{standard_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs_default}.{file_extension}"
                                zs_data = pickle.load(open(file_path, 'rb'))
                            else:
                                try:
                                    file_path = f"{path}/{standard_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs}.{file_extension}"
                                    zs_data = pickle.load(open(file_path, 'rb'))
                                except FileNotFoundError:
                                    file_path = f"{path}/{standard_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs_default}.{file_extension}"
                                    zs_data = pickle.load(open(file_path, 'rb'))
                                    print("Metrics loaded from " + file_path + " for condition " + str(cond) +
                                          ". Tried to load file " + file_name_zs + " unsuccessfully.")
                            if rsa:
                                file_path = f"{path}/{standard_path}/{zero_shot_path}/{cond}/{run_path}/{file_name}_{rsa_file_extension}.{file_extension}"
                                data = pickle.load(open(file_path, 'rb'))
                                # test acc
                                final_test_accs.append(data['final_test_acc'])
                                final_test_losses.append(data['final_test_loss'])
                                rsa_test_accs.append(data['rsa_test_acc'])
                                rsa_test_losses.append(data['rsa_test_loss'])
                        else:
                            if zero_shot_test_ds is None:
                                file_path = f"{path}/{length_cost_path}/{context_aware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs_default}.{file_extension}"
                                zs_data = pickle.load(open(file_path, 'rb'))
                            else:
                                try:
                                    file_path = f"{path}/{length_cost_path}/{context_aware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs}.{file_extension}"
                                    zs_data = pickle.load(open(file_path, 'rb'))
                                except FileNotFoundError:
                                    file_path = f"{path}/{length_cost_path}/{context_aware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs_default}.{file_extension}"
                                    zs_data = pickle.load(open(file_path, 'rb'))
                                    print("Metrics loaded from " + file_path + " for condition " + str(cond) +
                                          ". Tried to load file " + file_name_zs + " unsuccessfully.")
                            if rsa:
                                file_path = f"{path}/{length_cost_path}/{context_aware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name}_{rsa_file_extension}.{file_extension}"
                                data = pickle.load(open(file_path, 'rb'))
                                # test acc and rsa acc
                                if cond == 'specific':
                                    zs_specific_final_test_accs.append(data['final_test_acc'])
                                    zs_specific_final_test_losses.append(data['final_test_loss'])
                                    zs_specific_rsa_test_accs.append(data['rsa_test_acc'])
                                    zs_specific_rsa_test_losses.append(data['rsa_test_loss'])
                                else:
                                    zs_generic_final_test_accs.append(data['final_test_acc'])
                                    zs_generic_final_test_losses.append(data['final_test_loss'])
                                    zs_generic_rsa_test_accs.append(data['rsa_test_acc'])
                                    zs_generic_rsa_test_losses.append(data['rsa_test_loss'])

                        if zero_shot_test_ds is None:
                            # accuracies
                            lists = sorted(zs_data['metrics_train0'].items())
                            _, zs_train_acc = zip(*lists)
                            lists = sorted(zs_data['metrics_test0'].items())
                            _, zs_val_acc = zip(*lists)

                            # message lengths
                            lists = sorted(zs_data['metrics_train1'].items())
                            _, train_message_length = zip(*lists)
                            lists = sorted(zs_data['metrics_test1'].items())
                            _, val_message_length = zip(*lists)

                            if cond == 'specific':
                                zs_specific_train_accs.append(zs_train_acc)
                                zs_specific_val_accs.append(zs_val_acc)
                                zs_specific_test_accs.append(zs_data['final_test_acc'])
                                zs_specific_train_message_lengths.append(train_message_length)
                                zs_specific_val_message_lengths.append(val_message_length)
                            else:
                                zs_generic_train_accs.append(zs_train_acc)
                                zs_generic_val_accs.append(zs_val_acc)
                                zs_generic_test_accs.append(zs_data['final_test_acc'])
                                zs_generic_train_message_lengths.append(train_message_length)
                                zs_generic_val_message_lengths.append(val_message_length)
                        else:
                            if cond == 'specific':
                                zs_specific_test_accs.append(zs_data['final_test_acc'])
                            else:
                                zs_generic_test_accs.append(zs_data['final_test_acc'])

                    # zero-shot accuracy (context-unaware)
                    else:
                        if not length_cost:
                            if zero_shot_test_ds is None:
                                file_path = f"{path}/{context_unaware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs_default}.{file_extension}"
                                cu_zs_data = pickle.load(open(file_path, 'rb'))
                            else:
                                try:
                                    file_path = f"{path}/{context_unaware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs}.{file_extension}"
                                    cu_zs_data = pickle.load(open(file_path, 'rb'))
                                except FileNotFoundError:
                                    file_path = f"{path}/{context_unaware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs_default}.{file_extension}"
                                    cu_zs_data = pickle.load(open(file_path, 'rb'))
                                    print("Metrics loaded from " + file_path + " for condition " + str(cond) +
                                          ". Tried to load file " + file_name_zs + " unsuccessfully.")
                            if rsa:
                                file_path = f"{path}/{context_unaware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name}_{rsa_file_extension}.{file_extension}"
                                data = pickle.load(open(file_path, 'rb'))
                                # test acc and rsa acc
                                if cond == 'specific':
                                    zs_specific_final_test_accs.append(data['final_test_acc'])
                                    zs_specific_final_test_losses.append(data['final_test_loss'])
                                    zs_specific_rsa_test_accs.append(data['rsa_test_acc'])
                                    zs_specific_rsa_test_losses.append(data['rsa_test_loss'])
                                else:
                                    zs_generic_final_test_accs.append(data['final_test_acc'])
                                    zs_generic_final_test_losses.append(data['final_test_loss'])
                                    zs_generic_rsa_test_accs.append(data['rsa_test_acc'])
                                    zs_generic_rsa_test_losses.append(data['rsa_test_loss'])
                        else:
                            if zero_shot_test_ds is None:
                                file_path = f"{path}/{length_cost_path}/{context_unaware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs_default}.{file_extension}"
                                cu_zs_data = pickle.load(open(file_path, 'rb'))
                            else:
                                try:
                                    file_path = f"{path}/{length_cost_path}/{context_unaware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs}.{file_extension}"
                                    cu_zs_data = pickle.load(open(file_path, 'rb'))
                                except FileNotFoundError:
                                    file_path = f"{path}/{length_cost_path}/{context_unaware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name_zs_default}.{file_extension}"
                                    cu_zs_data = pickle.load(open(file_path, 'rb'))
                                    print("Metrics loaded from " + file_path + " for condition " + str(cond) +
                                          ". Tried to load file " + file_name_zs + " unsuccessfully.")
                            if rsa:
                                file_path = f"{path}/{length_cost_path}/{context_unaware_path}/{zero_shot_path}/{cond}/{run_path}/{file_name}_{rsa_file_extension}.{file_extension}"
                                data = pickle.load(open(file_path, 'rb'))
                                # test acc and rsa acc
                                if cond == 'specific':
                                    zs_specific_final_test_accs.append(data['final_test_acc'])
                                    zs_specific_final_test_losses.append(data['final_test_loss'])
                                    zs_specific_rsa_test_accs.append(data['rsa_test_acc'])
                                    zs_specific_rsa_test_losses.append(data['rsa_test_loss'])
                                else:
                                    zs_generic_final_test_accs.append(data['final_test_acc'])
                                    zs_generic_final_test_losses.append(data['final_test_loss'])
                                    zs_generic_rsa_test_accs.append(data['rsa_test_acc'])
                                    zs_generic_rsa_test_losses.append(data['rsa_test_loss'])

                        # if zero_shot_test_ds is None:
                        # accuracies
                        lists = sorted(cu_zs_data['metrics_train0'].items())
                        _, cu_zs_train_acc = zip(*lists)
                        lists = sorted(cu_zs_data['metrics_test0'].items())
                        _, cu_zs_val_acc = zip(*lists)

                        # message lengths
                        lists = sorted(cu_zs_data['metrics_train1'].items())
                        _, train_message_length = zip(*lists)
                        lists = sorted(cu_zs_data['metrics_test1'].items())
                        _, val_message_length = zip(*lists)

                        if cond == 'specific':
                            cu_zs_specific_train_accs.append(cu_zs_train_acc)
                            cu_zs_specific_val_accs.append(cu_zs_val_acc)
                            cu_zs_specific_test_accs.append(cu_zs_data['final_test_acc'])
                            cu_zs_specific_train_message_lengths.append(train_message_length)
                            cu_zs_specific_val_message_lengths.append(val_message_length)
                        else:
                            cu_zs_generic_train_accs.append(cu_zs_train_acc)
                            cu_zs_generic_val_accs.append(cu_zs_val_acc)
                            cu_zs_generic_test_accs.append(cu_zs_data['final_test_acc'])
                            cu_zs_generic_train_message_lengths.append(train_message_length)
                            cu_zs_generic_val_message_lengths.append(val_message_length)


        if not context_unaware and not zero_shot:
            if rsa_test is None:
                result_dict['train_acc'].append(train_accs)
                result_dict['val_acc'].append(val_accs)
                result_dict['test_acc'].append(test_accs)
                result_dict['train_message_lengths'].append(train_message_lengths)
                result_dict['val_message_lengths'].append(val_message_lengths)
            else:
                result_dict['final_test_acc'].append(final_test_accs)
                result_dict['final_test_loss'].append(final_test_losses)
                result_dict['rsa_test_acc'].append(rsa_test_accs)
                result_dict['rsa_test_loss'].append(rsa_test_losses)
        elif context_unaware and not zero_shot:
            if rsa_test is None:
                result_dict['cu_train_acc'].append(cu_train_accs)
                result_dict['cu_val_acc'].append(cu_val_accs)
                result_dict['cu_test_acc'].append(cu_test_accs)
                result_dict['cu_train_message_lengths'].append(cu_train_message_lengths)
                result_dict['cu_val_message_lengths'].append(cu_val_message_lengths)
            else:
                result_dict['final_test_acc'].append(final_test_accs)
                result_dict['final_test_loss'].append(final_test_losses)
                result_dict['rsa_test_acc'].append(rsa_test_accs)
                result_dict['rsa_test_loss'].append(rsa_test_losses)
                result_dict['rsa_test_gen_utt_acc'].append(rsa_test_gen_utt_accs)
                result_dict['rsa_test_gen_utt_loss'].append(rsa_test_gen_utt_losses)
        elif not context_unaware and zero_shot:
            result_dict['zs_specific_train_acc'].append(zs_specific_train_accs)
            result_dict['zs_specific_val_acc'].append(zs_specific_val_accs)
            result_dict['zs_specific_test_acc'].append(zs_specific_test_accs)
            result_dict['zs_specific_train_message_length'].append(zs_specific_train_message_lengths)
            result_dict['zs_specific_val_message_length'].append(zs_specific_val_message_lengths)
            result_dict['zs_generic_train_acc'].append(zs_generic_train_accs)
            result_dict['zs_generic_val_acc'].append(zs_generic_val_accs)
            result_dict['zs_generic_test_acc'].append(zs_generic_test_accs)
            result_dict['zs_generic_train_message_length'].append(zs_generic_train_message_lengths)
            result_dict['zs_generic_val_message_length'].append(zs_generic_val_message_lengths)
            if rsa:
                result_dict['zs_specific_final_test_acc'].append(zs_specific_final_test_accs)
                result_dict['zs_specific_final_test_loss'].append(zs_specific_final_test_losses)
                result_dict['zs_specific_rsa_test_acc'].append(zs_specific_rsa_test_accs)
                result_dict['zs_specific_rsa_test_loss'].append(zs_specific_rsa_test_losses)
                result_dict['zs_generic_final_test_acc'].append(zs_generic_final_test_accs)
                result_dict['zs_generic_final_test_loss'].append(zs_generic_final_test_losses)
                result_dict['zs_generic_rsa_test_acc'].append(zs_generic_rsa_test_accs)
                result_dict['zs_generic_rsa_test_loss'].append(zs_generic_rsa_test_losses)
        elif context_unaware and zero_shot:
            result_dict['cu_zs_specific_train_acc'].append(cu_zs_specific_train_accs)
            result_dict['cu_zs_specific_val_acc'].append(cu_zs_specific_val_accs)
            result_dict['cu_zs_specific_test_acc'].append(cu_zs_specific_test_accs)
            result_dict['cu_zs_specific_train_message_length'].append(cu_zs_specific_train_message_lengths)
            result_dict['cu_zs_specific_val_message_length'].append(cu_zs_specific_val_message_lengths)
            result_dict['cu_zs_generic_train_acc'].append(cu_zs_generic_train_accs)
            result_dict['cu_zs_generic_val_acc'].append(cu_zs_generic_val_accs)
            result_dict['cu_zs_generic_test_acc'].append(cu_zs_generic_test_accs)
            result_dict['cu_zs_generic_train_message_length'].append(cu_zs_generic_train_message_lengths)
            result_dict['cu_zs_generic_val_message_length'].append(cu_zs_generic_val_message_lengths)
            if rsa:
                result_dict['zs_specific_final_test_acc'].append(zs_specific_final_test_accs)
                result_dict['zs_specific_final_test_loss'].append(zs_specific_final_test_losses)
                result_dict['zs_specific_rsa_test_acc'].append(zs_specific_rsa_test_accs)
                result_dict['zs_specific_rsa_test_loss'].append(zs_specific_rsa_test_losses)
                result_dict['zs_generic_final_test_acc'].append(zs_generic_final_test_accs)
                result_dict['zs_generic_final_test_loss'].append(zs_generic_final_test_losses)
                result_dict['zs_generic_rsa_test_acc'].append(zs_generic_rsa_test_accs)
                result_dict['zs_generic_rsa_test_loss'].append(zs_generic_rsa_test_losses)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict


def load_accuracies_rsa_zero_shot(all_paths, result_dict=None, n_runs=5, setting='standard/zero_shot/', zs_test='specific',
                                  granularity='fine'):
    """loads accuracies for the specific zero-shot condition and a specified granularity of the dataset"""
    add_test_acc = False
    if not result_dict:
        result_dict = {'rsa_test_acc': [], 'final_test_acc': []}
        add_test_acc = True

    for path_idx, path in enumerate(all_paths):
        rsa_test_accs, final_test_accs = [], []

        for run in range(n_runs):
            path_to_run = path + '/' + setting + zs_test + '/' + str(run) + '/'
            if zs_test == 'specific':
                file_path = path_to_run + 'loss_and_metrics_test_' + granularity + '.pkl'
            else:
                file_path = path_to_run + 'loss_and_metrics_test.pkl'
            # with open(os.path.join(path_to_run, 'params.pkl'), 'rb') as input_file:
            with open(os.path.join(file_path), 'rb') as input_file:
                data = pickle.load(input_file)
                rsa_test_accs.append(data['rsa_test_acc'])
                if add_test_acc:
                    final_test_accs.append(data['final_test_acc'])
        result_dict['rsa_test_acc'].append(rsa_test_accs)
        if add_test_acc:
            result_dict['final_test_acc'].append(final_test_accs)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict

def load_entropies(all_paths, n_runs=5, context_unaware=False, length_cost=False, rsa=False, rsa_test=None,
                   sampled_context=False, test_interactions=False, test_mode=None, hierarchical=False,
                   shared_context=False, verbose=False, split_by_attribute=False):
    """ loads all entropy scores into a dictionary"""

    if sampled_context:
        path_sc = '/sampled_context'
    else:
        path_sc = ''
    if length_cost:
        if context_unaware:
            setting = 'length_cost/context_unaware'
        elif not context_unaware:
            setting = 'length_cost/context_aware'
    else:
        if context_unaware:
               setting = 'context_unaware' + path_sc
        else:
            setting = 'standard' + path_sc
    if hierarchical:
        setting = setting + '/hierarchical'
    if shared_context and not split_by_attribute:
        setting = setting + '/shared_context'
    if split_by_attribute:
        setting = setting + '/split_by_attribute'

    if rsa:
        rsa_file_extension = '_rsa_' + rsa_test
    else:
        rsa_file_extension = ''

    if test_interactions:
        test_file_extension = '_' + test_mode
    else:
        test_file_extension = ''

    result_dict = {'NMI': [], 'effectiveness': [], 'consistency': [],
                   'NMI_hierarchical': [], 'effectiveness_hierarchical': [], 'consistency_hierarchical': [],
                   'NMI_context_dep': [], 'effectiveness_context_dep': [], 'consistency_context_dep': [],
                   'NMI_concept_x_context': [], 'effectiveness_concept_x_context': [],
                   'consistency_concept_x_context': []}

    for path_idx, path in enumerate(all_paths):

        NMIs, effectiveness_scores, consistency_scores = [], [], []
        NMIs_hierarchical, effectiveness_scores_hierarchical, consistency_scores_hierarchical = [], [], []
        NMIs_context_dep, effectiveness_scores_context_dep, consistency_scores_context_dep = [], [], []
        NMIs_conc_x_cont, effectiveness_conc_x_cont, consistency_conc_x_cont = [], [], []

        for run in range(n_runs):
            standard_path = path + '/' + setting + '/' + str(run) + '/'
            data = pickle.load(open(standard_path + 'entropy_scores' + rsa_file_extension + test_file_extension + '.pkl', 'rb'))
            if verbose:
                print("Entropy scores loaded from:", standard_path + 'entropy_scores' + rsa_file_extension + test_file_extension + '.pkl')
            NMIs.append(data['normalized_mutual_info'])
            effectiveness_scores.append(data['effectiveness'])
            consistency_scores.append(data['consistency'])
            NMIs_hierarchical.append(data['normalized_mutual_info_hierarchical'])
            effectiveness_scores_hierarchical.append(data['effectiveness_hierarchical'])
            consistency_scores_hierarchical.append(data['consistency_hierarchical'])
            NMIs_context_dep.append(data['normalized_mutual_info_context_dep'])
            effectiveness_scores_context_dep.append(data['effectiveness_context_dep'])
            consistency_scores_context_dep.append(data['consistency_context_dep'])
            NMIs_conc_x_cont.append(data['normalized_mutual_info_concept_x_context'])
            effectiveness_conc_x_cont.append(data['effectiveness_concept_x_context'])
            consistency_conc_x_cont.append(data['consistency_concept_x_context'])

        result_dict['NMI'].append(NMIs)
        result_dict['consistency'].append(consistency_scores)
        result_dict['effectiveness'].append(effectiveness_scores)
        result_dict['NMI_hierarchical'].append(NMIs_hierarchical)
        result_dict['consistency_hierarchical'].append(consistency_scores_hierarchical)
        result_dict['effectiveness_hierarchical'].append(effectiveness_scores_hierarchical)
        result_dict['NMI_context_dep'].append(NMIs_context_dep)
        result_dict['consistency_context_dep'].append(consistency_scores_context_dep)
        result_dict['effectiveness_context_dep'].append(effectiveness_scores_context_dep)
        result_dict['NMI_concept_x_context'].append(NMIs_conc_x_cont)
        result_dict['consistency_concept_x_context'].append(consistency_conc_x_cont)
        result_dict['effectiveness_concept_x_context'].append(effectiveness_conc_x_cont)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict


def load_entropies_zero_shot(all_paths, n_runs=5, context_unaware=False, length_cost=False, test_interactions=False,
                             zero_shot_test_ds='test', rsa=False, rsa_test=None, verbose=False):
    """ loads all entropy scores into a dictionary"""

    if length_cost:
        if context_unaware:
            setting = 'length_cost/context_unaware'
        elif not context_unaware:
            setting = 'length_cost/context_aware'
    else:
        if context_unaware:
            setting = 'context_unaware'
        else:
            setting = 'standard'

    setting = str(setting + '/' + 'zero_shot')

    if rsa:
        rsa_file_extension = '_rsa_' + rsa_test
    else:
        rsa_file_extension = ''

    result_dict_specific = {'NMI': [], 'effectiveness': [], 'consistency': [],
                            'NMI_hierarchical': [], 'effectiveness_hierarchical': [], 'consistency_hierarchical': [],
                            'NMI_context_dep': [], 'effectiveness_context_dep': [], 'consistency_context_dep': [],
                            'NMI_concept_x_context': [], 'effectiveness_concept_x_context': [],
                            'consistency_concept_x_context': []}
    result_dict_generic = copy.deepcopy(result_dict_specific)

    for cond in ['specific', 'generic']:

        for path_idx, path in enumerate(all_paths):

            NMIs, effectiveness_scores, consistency_scores = [], [], []
            NMIs_hierarchical, effectiveness_scores_hierarchical, consistency_scores_hierarchical = [], [], []
            NMIs_context_dep, effectiveness_scores_context_dep, consistency_scores_context_dep = [], [], []
            NMIs_conc_x_cont, effectiveness_conc_x_cont, consistency_conc_x_cont = [], [], []

            for run in range(n_runs):
                standard_path = path + '/' + setting + '/' + cond + '/' + str(run) + '/'
                if not test_interactions:
                    data = pickle.load(open(standard_path + 'entropy_scores.pkl', 'rb'))
                    if verbose:
                        print("Entropy scores loaded from:", standard_path + 'entropy_scores.pkl')
                else:
                    data = pickle.load(open(standard_path + 'entropy_scores_' + zero_shot_test_ds + '.pkl', 'rb'))
                    if verbose:
                        print("Entropy scores loaded from:", standard_path + 'entropy_scores_' + zero_shot_test_ds + '.pkl')
                if rsa:
                    data = pickle.load(open(standard_path + 'entropy_scores' + rsa_file_extension + '.pkl', 'rb'))
                    if verbose:
                        print("Entropy scores loaded from:", standard_path + 'entropy_scores' + rsa_file_extension + '.pkl')
                NMIs.append(data['normalized_mutual_info'])
                effectiveness_scores.append(data['effectiveness'])
                consistency_scores.append(data['consistency'])
                NMIs_hierarchical.append(data['normalized_mutual_info_hierarchical'])
                effectiveness_scores_hierarchical.append(data['effectiveness_hierarchical'])
                consistency_scores_hierarchical.append(data['consistency_hierarchical'])
                NMIs_context_dep.append(data['normalized_mutual_info_context_dep'])
                effectiveness_scores_context_dep.append(data['effectiveness_context_dep'])
                consistency_scores_context_dep.append(data['consistency_context_dep'])
                NMIs_conc_x_cont.append(data['normalized_mutual_info_concept_x_context'])
                effectiveness_conc_x_cont.append(data['effectiveness_concept_x_context'])
                consistency_conc_x_cont.append(data['consistency_concept_x_context'])

            if cond == 'specific':
                result_dict_specific['NMI'].append(NMIs)
                result_dict_specific['consistency'].append(consistency_scores)
                result_dict_specific['effectiveness'].append(effectiveness_scores)
                result_dict_specific['NMI_hierarchical'].append(NMIs_hierarchical)
                result_dict_specific['consistency_hierarchical'].append(consistency_scores_hierarchical)
                result_dict_specific['effectiveness_hierarchical'].append(effectiveness_scores_hierarchical)
                result_dict_specific['NMI_context_dep'].append(NMIs_context_dep)
                result_dict_specific['consistency_context_dep'].append(consistency_scores_context_dep)
                result_dict_specific['effectiveness_context_dep'].append(effectiveness_scores_context_dep)
                result_dict_specific['NMI_concept_x_context'].append(NMIs_conc_x_cont)
                result_dict_specific['consistency_concept_x_context'].append(consistency_conc_x_cont)
                result_dict_specific['effectiveness_concept_x_context'].append(effectiveness_conc_x_cont)
            else:
                result_dict_generic['NMI'].append(NMIs)
                result_dict_generic['consistency'].append(consistency_scores)
                result_dict_generic['effectiveness'].append(effectiveness_scores)
                result_dict_generic['NMI_hierarchical'].append(NMIs_hierarchical)
                result_dict_generic['consistency_hierarchical'].append(consistency_scores_hierarchical)
                result_dict_generic['effectiveness_hierarchical'].append(effectiveness_scores_hierarchical)
                result_dict_generic['NMI_context_dep'].append(NMIs_context_dep)
                result_dict_generic['consistency_context_dep'].append(consistency_scores_context_dep)
                result_dict_generic['effectiveness_context_dep'].append(effectiveness_scores_context_dep)
                result_dict_generic['NMI_concept_x_context'].append(NMIs_conc_x_cont)
                result_dict_generic['consistency_concept_x_context'].append(consistency_conc_x_cont)
                result_dict_generic['effectiveness_concept_x_context'].append(effectiveness_conc_x_cont)

    for key in result_dict_specific.keys():
        result_dict_specific[key] = np.array(result_dict_specific[key])
    for key in result_dict_generic.keys():
        result_dict_generic[key] = np.array(result_dict_generic[key])

    return result_dict_specific, result_dict_generic



# Mu and goodman:

def load_accuracies_mu_and_goodman(all_paths, n_runs=5, n_epochs=300, val_steps=10, zero_shot=True):
    """ loads all mu and goodman accuracies into a dictionary, val_steps should be set to the same as val_frequency
    during training
    """
    result_dict = {'zs_specific_test_acc': [],
                   'zs_generic_test_acc': []}

    for path_idx, path in enumerate(all_paths):

        zs_specific_test_accs = []
        zs_generic_test_accs = []

        for run in range(n_runs):

            mu_and_goodman_path = path + '/mu_and_goodman/zero_shot/'

            if zero_shot:
                for cond in ['specific', 'generic']:

                    # zero shot accuracy (standard)
                    zs_data = pickle.load(
                        open(mu_and_goodman_path + str(cond) + '/' + str(run) + '/loss_and_metrics.pkl', 'rb'))
                    if cond == 'specific':
                        zs_specific_test_accs.append(zs_data['final_test_acc'])
                    else:
                        zs_generic_test_accs.append(zs_data['final_test_acc'])

        if zero_shot:
            result_dict['zs_specific_test_acc'].append(zs_specific_test_accs)
            result_dict['zs_generic_test_acc'].append(zs_generic_test_accs)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict
