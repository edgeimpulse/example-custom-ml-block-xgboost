import sys, os, argparse
import numpy as np
from sklearn import metrics
import xgboost as xgb

RANDOM_SEED = 1
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Edge Impulse XGBOOST training scripts')

parser.add_argument('--data-directory', type=str, required=True,
                    help='Where to read the data from')
parser.add_argument('--out-directory', type=str, required=True,
                    help='Where to write the data')

parser.add_argument('--learning-rate', type=float, required=False, default=0.3,
                    help='Step size shrinkage used in update to prevents overfitting.')
parser.add_argument('--max-depth', type=int, required=False, default=6,
                    help='Maximum depth')
parser.add_argument('--num-boost-round', type=int, required=False, default=10,
                    help='Number of boosting iterations.')

parser.add_argument('--l2', type=float, required=False, default=1.0,
                    help='L2 regularization term on weights. Increasing this value will make model more conservative.')
parser.add_argument('--l1', type=float, required=False, default=0.0,
                    help='L1 regularization term on weights. Increasing this value will make model more conservative.')
parser.add_argument('--max-leaves', type=int, required=False,
                    help='Maximum number of nodes to be added.')
parser.add_argument('--min-child-weight', type=float, required=False, default=0.0,
                    help='Minimum sum of instance weight (hessian) needed in a child.')
parser.add_argument('--subsample', type=float, required=False, default=1.0,
                    help='Subsample ratio of the training instances.')

args, unknown = parser.parse_known_args()

out_directory = args.out_directory

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'))
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'))
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

num_classes = Y_train.shape[1]

# sparse representation of the labels
Y_train = np.argmax(Y_train, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print('num classes: ' + str(num_classes))
print('mode: classifier')

params = None
if num_classes == 2:
    params = {
        "objective": "binary:logistic"
    }
else:
    params =  {
        'objective': 'multi:softmax',
        'num_class': num_classes
    }

params['learning_rate'] = args.learning_rate
params['max_depth'] = args.max_depth
params['lambda'] = args.l2
params['alpha'] = args.l1
params['min_child_weight'] = args.min_child_weight
params['subsample'] = args.subsample

print('params:', params)
print('')
print('Training XGBoost random forest...')

D_train = xgb.DMatrix(X_train, Y_train)
D_valid = xgb.DMatrix(X_test, Y_test)
clf = xgb.train(
    params,
    D_train,
    evals=[(D_train, 'Train'), (D_valid, 'Valid')],
    num_boost_round=args.num_boost_round,
    verbose_eval=True)

print('Training XGBoost random forest OK')
print(' ')
print('Calculating XGBoost random forest accuracy...')

num_correct = 0
for idx in range(len(Y_test)):
    pred = clf.predict(xgb.DMatrix(X_test[idx].reshape(1, -1)))
    if num_classes == 2:
        if Y_test[idx] == int(round(pred[0])):
            num_correct += 1
    else:
        if Y_test[idx] == pred[0]:
            num_correct += 1
print(f'Accuracy (validation set): {num_correct / len(Y_test)}')

print('')
print('Saving XGBoost model...')
file_xgb = os.path.join(args.out_directory, 'model.json')
clf.save_model(file_xgb)
print('Saving XGBoost model OK')
