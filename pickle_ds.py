import torch
from dataset import DataSet
import argparse
import os

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)

parser = argparse.ArgumentParser()

parser.add_argument('--dimensions', nargs="*", type=int, default=[3, 3, 3],
                    help='Number of features for every perceptual dimension')
parser.add_argument('--game_size', type=int, default=10,
                    help='Number of target/distractor objects')
parser.add_argument('--scaling_factor', type=int, default=10,
                    help='Scaling factor for dataset generation.')
parser.add_argument('--zero_shot', type=bool, default=False,
                    help='Set to True if zero-shot datasets should be generated.')
parser.add_argument("--zero_shot_test", type=str, default=None,
                    help='Set to either "specific" or "generic" for different zero shot test conditions.')
parser.add_argument("--save", type=bool, default=True)
parser.add_argument('--sample_context', type=bool, default=False,
                    help="If true, sample context condition instead of generating all possible context condition for "
                         "each concept.")
parser.add_argument('--granularity', type=str, default='mixed',
                    help='Granularity of the context. Possible values: mixed, coarse and fine')
parser.add_argument('--shared_context', type=bool, default=False,
                    help="If true, context is generated with specific shared attributes instead of all possible.")
parser.add_argument('--split_by_attribute', type=bool, default=False,
                    help= 'Use for generating dataset with highly discriminative attribute.')
parser.add_argument('--percentage_a', type=float, default=0.8,
                    help='For split_by_attribute: proportion of the subset A in train+val dataset.')

args = parser.parse_args()

# prepare folder for saving
if not os.path.exists('data/'):
    os.makedirs('data/')

# prepare appendix for dataset name if sample_context or shared context etc.
sample = ''
if args.sample_context:
    sample = sample + '_context_sampled'
if args.shared_context and not args.split_by_attribute:
    sample = sample + '_shared_context'
if args.split_by_attribute:
    sample = sample + '_split_by_attribute'

# for normal dataset (not zero-shot)
if not args.zero_shot:
    data_set = DataSet(args.dimensions,
                       game_size=args.game_size,
                       scaling_factor=args.scaling_factor,
                       device='cpu',
                       sample_context=args.sample_context,
                       granularity=args.granularity,
                       shared_context=args.shared_context,
                       split_by_attribute=args.split_by_attribute,
                       percentage_a=args.percentage_a)

    if data_set.granularity == 'mixed' or data_set.granularity == None:
        path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')' + sample + '_sf' +
                str(args.scaling_factor) + '.ds')
    else:
        path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')' + sample + '_granularity_'
                + str(args.granularity) + '_sf' + str(args.scaling_factor) + '.ds')

    if args.save:
        with open(path, "wb") as f:
            torch.save(data_set, f)
        print("Data set is saved as: " + path)

# for zero-shot datasets
else:
    if args.zero_shot_test is None:
        for cond in ['generic', 'specific']:
            data_set = DataSet(args.dimensions,
                               game_size=args.game_size,
                               scaling_factor=args.scaling_factor,
                               testing=True,
                               device='cpu',
                               sample_context=args.sample_context,
                               zero_shot=True,
                               zero_shot_test=cond,
                               granularity=args.granularity)
            if data_set.granularity == 'mixed' or data_set.granularity == None:
                path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')' + sample + '_' +
                        str(cond) + '_sf' + str(args.scaling_factor) + '.ds')
            else:
                path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')' + sample + '_' +
                        str(cond) + '_granularity_' + str(args.granularity) + '_sf' + str(args.scaling_factor) + '.ds')

    else:
        data_set = DataSet(args.dimensions,
                           game_size=args.game_size,
                           scaling_factor=args.scaling_factor,
                           testing=True,
                           device='cpu',
                           sample_context=args.sample_context,
                           zero_shot=True,
                           zero_shot_test=args.zero_shot_test,
                           granularity=args.granularity)

        if data_set.granularity == 'mixed' or data_set.granularity == None:
            path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')' + sample + '_' +
                    str(args.zero_shot_test) + '_sf' + str(args.scaling_factor) + '.ds')
        else:
            path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')' + sample + '_' +
                    str(args.zero_shot_test) + '_granularity_' + str(args.granularity) + '_sf' + str(args.scaling_factor)
                    + '.ds')

        if args.save:
            with open(path, "wb") as f:
                torch.save(data_set, f)
            print("Data set is saved as: " + path)
