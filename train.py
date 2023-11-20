import sys, os, argparse
import lightgbm as lgb
import numpy as np
import sys, os, signal, random, time, argparse
import logging, threading
from sklearn import metrics
import xgboost as xgb

sys.path.append('./resources/libraries')
import ei_tensorflow.training
from ei_shared.parse_train_input import parse_train_input, parse_input_shape


RANDOM_SEED = 1
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Edge Impulse training scripts')
parser.add_argument('--info-file', type=str, required=False,
                    help='train_input.json file with info about classes and input shape',
                    default=os.path.join(dir_path, 'train_input.json'))
parser.add_argument('--data-directory', type=str, required=True,
                    help='Where to read the data from')
parser.add_argument('--out-directory', type=str, required=True,
                    help='Where to write the data')

parser.add_argument('--learning-rate', type=float, required=False,
                    help='Step size shrinkage used in update to prevents overfitting.')
parser.add_argument('--max-leaves', type=int, required=False,
                    help='Maximum number of nodes to be added.')
parser.add_argument('--num-parallel-tree', type=int, required=False,
                    help='Number of parallel trees constructed during each iteration.')
parser.add_argument('--max-depth', type=int, required=False,
                    help='Maximum depth')
parser.add_argument('--num-boost-round', type=int, required=False,
                    help='Number of boosting iterations.')

args, unknown = parser.parse_known_args()

# Info about the training pipeline (inputs / shapes / modes etc.)
if not os.path.exists(args.info_file):
    print('Info file', args.info_file, 'does not exist')
    exit(1)

input = parse_train_input(args.info_file)

BEST_MODEL_PATH = os.path.join(os.sep, 'tmp', 'best_model.tf' if input.akidaModel else 'best_model.hdf5')

# Information about the data and input:
# The shape of the model's input (which may be different from the shape of the data)
MODEL_INPUT_SHAPE = parse_input_shape(input.inputShapeString)
# The length of the model's input, used to determine the reshape inside the model
MODEL_INPUT_LENGTH = MODEL_INPUT_SHAPE[0]
MAX_TRAINING_TIME_S = input.maxTrainingTimeSeconds

class XGBLogging(xgb.callback.TrainingCallback):
    def __init__(self, epoch_log_interval=100):
        self.epoch_log_interval = epoch_log_interval

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.epoch_log_interval == 0:
            for data, metric in evals_log.items():
                metrics = list(metric.keys())
                metrics_str = ""
                for m_key in metrics:
                    metrics_str = metrics_str + f"{m_key}: {metric[m_key][-1]}"
                print(f"Epoch: {epoch}, {data}: {metrics_str}")
        return False


def exit_gracefully(signum, frame):
    print("")
    print("Terminated by user", flush=True)
    time.sleep(0.2)
    sys.exit(1)

def main_function():
    """This function is used to avoid contaminating the global scope"""

    train_dataset, validation_dataset, samples_dataset, X_train, X_test, Y_train, Y_test, has_samples, X_samples, Y_samples = ei_tensorflow.training.get_dataset_from_folder(
        input, args.data_directory, RANDOM_SEED, None, MODEL_INPUT_SHAPE, None
    )

    print('')
    print('Training XGBoost model...')

    if input.mode == 'classification':
        Y_train = np.argmax(Y_train, axis=1)
        Y_test = np.argmax(Y_test, axis=1)

    max_depth = args.max_depth or 6
    num_parallel_tree = args.num_parallel_tree or 1
    max_leaves = args.max_leaves or 0
    learning_rate = args.learning_rate or 0.3
    num_boost_round = args.num_boost_round or 10

    num_features = MODEL_INPUT_SHAPE[0]
    num_classes = len(input.classes)

    print('Max. depth: ' + str(max_depth))
    print('Num. parallel tree: ' + str(num_parallel_tree))
    print('Max. leaves: ' + str(max_leaves))
    print('Learning rate: ' + str(learning_rate))
    print('Num. boost round: ' + str(num_boost_round))
    print('')
    print('num features: ' + str(num_features))
    print('num classes: ' + str(num_classes))
    print('mode: ' + str(input.mode))

    params = None
    if input.mode == 'regression':
        params = {
            "objective": "reg:squarederror"
        }
    else:
        if num_classes == 2:
            params = {
                "objective": "binary:logistic"
            }
        else:
            params =  {
                'objective': 'multi:softmax',
                'num_class': num_classes
            }

    D_train = xgb.DMatrix(X_train, Y_train)
    D_valid = xgb.DMatrix(X_test, Y_test)
    clf = xgb.train(
        params,
        D_train,
        evals=[(D_train, 'Train'), (D_valid, 'Valid')],
        num_boost_round=num_boost_round,
        verbose_eval=True)

    print(' ')
    print('Calculating XGBoost random forest accuracy...')

    if input.mode == 'regression':
        predicted_y = clf.predict(xgb.DMatrix(X_test))
        print('r^2: ' + str(metrics.r2_score(Y_test, predicted_y)))
        print('mse: ' + str(metrics.mean_squared_error(Y_test, predicted_y)))
        print('log(mse): ' + str(metrics.mean_squared_log_error(Y_test, predicted_y)))
    else:
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

    print('Saving XGBoost model...')
    file_lgbm = os.path.join(args.out_directory, 'model.json')
    clf.save_model(file_lgbm)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    main_function()