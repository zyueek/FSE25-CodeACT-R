from __future__ import annotations

import argparse
import copy
import itertools
import os
import pickle
import random
import time
import ast
import json
import warnings

import pandas as pd
from keras.callbacks import CSVLogger
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.regularizers import l2
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
import seaborn as sns
import keras_tuner
import numpy as np
import tensorflow
import tensorflow as tf
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# import config as C
from utils.gpu_selection import configure_gpu, get_gpu_dict
from utils.gpu_selection import select_gpu
from utils.utils import load_data, load_data_stim, train_test_split_by_stim
from utils.utils import train_test_split_by_code_snippets
from utils.utils import train_test_split_by_participants

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1)
parser.add_argument(
    '--problem-setting', type=str, required=True,
    choices=['accuracy', 'subjective_difficulty', 'subjective_difficulty_score'],
)
parser.add_argument(
    '--split', type=str, required=True,
    choices=['subject', 'code-snippet'],
)
parser.add_argument(
    '--mode', type=str, required=True,
    choices=['bimodal', 'fixations', 'code'],
)
parser.add_argument('--less-fold', action='store_true')
parser.add_argument(
    '--split-file', type=str, required=False,
)
parser.add_argument(
    '--fold-offset', type=int, required=False,
)
parser.add_argument(
    '--simulation', type=str, required=True,
)
parser.add_argument(
    '--seed', type=str, required=True,
)
parser.add_argument(
    '--output', type=str, required=True,
)


args = parser.parse_args()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

EMBED_SIZE = 768
MAX_CS_EMBED_SIZE = 166
def help_f1_score(y_true, y_pred):
    # Convert tensors to numpy arrays
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    # Convert probabilities to binary predictions (0 or 1)
    y_pred = np.round(y_pred)
    return f1_score(y_true, y_pred)
def f1_metric(y_true, y_pred):
    return tf.py_function(help_f1_score, (y_true, y_pred), tf.double)
def help_roc_auc(y_true, y_pred):
#    print(y_true,y_pred)
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)
def help_recall_score(y_true, y_pred):
    # Convert tensors to numpy arrays
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    # Convert probabilities to binary predictions (0 or 1)
    y_pred = np.round(y_pred)
    return recall_score(y_true, y_pred)
def recall_metric(y_true, y_pred):
    return tf.py_function(help_recall_score, (y_true, y_pred), tf.double)
def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)

class AttentionLayer(tensorflow.keras.layers.Layer):
    def __init__(self, data, window_size, lm="codebert", **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.data = tf.Variable(data, dtype=tf.float64, trainable=False)
        self.window_size = window_size
        self.total_window_size = 2 * window_size + 1
        self.lm_idx = 0 if lm == "codebert" else 1
        self.w = self.add_weight(
            name='attention_weight',
            shape=(self.total_window_size,EMBED_SIZE), # shape=(self.total_window_size,),
            dtype=tf.float64,
            initializer='random_normal',
            trainable=True)

    def call(self, inputs):
        inputs, code_snippet_id = inputs
        inputs = tf.cast(inputs, tf.int32)
        code_snippet_id = tf.cast(code_snippet_id, tf.int32)

        # expand dims for proper broadcasting
        indices = tf.range(-self.window_size, self.window_size + 1) + tf.expand_dims(inputs, -1)
        end_clip = MAX_CS_EMBED_SIZE - 1
        indices = tf.clip_by_value(indices, 0, end_clip)

        # bring code_snippet_id into correct shape for gather_nd indices tensor
        expanded_code_snippet_id = tf.expand_dims(code_snippet_id, axis=-1)  # shape: [?, 1, 1]
        tiled_code_snippet_id = tf.tile(expanded_code_snippet_id,
                                        [1, 128, self.total_window_size])  # shape: [?, 1126, 1/3/5]

        lm_idx_tensor = tf.fill(tf.shape(indices), self.lm_idx)


        # Stack the tensors to get [code_snippet_id, lm_idx, indices_i] for each indices_i in indices
        indices_tensor = tf.stack([tiled_code_snippet_id, indices, lm_idx_tensor], axis=-1)

        # gather LLM embeddings in window_size
        gathered_data = tf.gather_nd(params=self.data, indices=indices_tensor)

        # multipy with attention weights and then reduce sum over the window size axis.
        output = gathered_data * self.w

        # Broadcast self.w to match the shape of gathered_data
        #expanded_w = tf.tile(tf.expand_dims(self.w, axis=1), [1, EMBED_SIZE])

        # Multiply the gathered data with the expanded weights
        #output = gathered_data * expanded_w

        output = tf.reduce_sum(output, axis=2)  # Sum over the window size axis

        return output


# main function to execute nested CV
def main(test_stim,X):
    start_time = time.time()


    configure_gpu(select_gpu(10000))
    print(get_gpu_dict())
    # load data
 #   print(X)
    X['difficulty'] = (
            X['difficulty'] > X['difficulty'].mean()
    ).astype(int)
    seed = int(args.seed)
#    X = X.sample(frac=1,random_state=seed).reset_index(drop=True)
#    print(X)
#    print(X)
    # get train_test_split (s.t. 1 cs is always left out from training entirely)
#    print(X)
    code_snippet_ids = set(X['stim'])
    participant_ids = set(X['participant'])

    # check if manual split file is provided
    if args.split_file is not None:
        print(f"Loading existing split {args.split_file}")
        # Load the file
        with open(f'{args.split_file}', 'r') as file:
            data = file.read().replace('\n', '')

        # Convert string to list
        data_list = ast.literal_eval(data)

        # Convert list to numpy array
        train_test_splits = np.array(data_list)
    # split by code-snippet
    elif args.split == 'code-snippet':
        train_test_splits = train_test_split_by_stim(
            test_stim,
            code_snippet_ids,
            X,
            args.problem_setting,
            True,
        )
    # split by subject
    elif args.split == 'subject':
        train_test_splits = train_test_split_by_participants(
            participant_ids,
            X,
            args.problem_setting,
            args.less_fold,
        )
    else:
        raise NotImplementedError(f'{args.split=}')
#    print(X['difficulty'])
    # select what to predict (i.e. task outcome => accuracy, or subj_difficulty)
    if args.problem_setting == 'accuracy':
        y = X.pop('accuracy')
        y = y.to_numpy()
        X.pop('subjective_difficulty')
    elif args.problem_setting == 'subjective_difficulty':
        problem_setting = X.pop('difficulty')
#        print(problem_setting)
        problem_setting_mean = np.mean(problem_setting)
        y = np.array(problem_setting > problem_setting_mean, dtype=int)
#        print(y)
#        X.pop('accuracy')
    elif args.problem_setting == 'subjective_difficulty_score':
        y = X.pop('difficulty')
        y = y.to_numpy()
        y = (y - 1) / 4 # normalize scale 1-5 to 0-1
    else:
        raise NotImplementedError(f'{args.problem_setting=}')

#    print(y)
    if args.fold_offset is None:
        args.fold_offset = 0

    X.pop('participant')
    X_code_snippet_ids = X.pop('stim')
#    print(X)
    # Open the CS_ID to CS_IDX mapping
    with open(f"../processed_data/stim_id_mapping.json", 'r') as f:
        CS_ID_to_IDX_MAP = json.load(f)

    # get code_snippet embedding IDXs (instead of string IDs)
    X_code_snippet_IDXs = [CS_ID_to_IDX_MAP.get(CS_ID) for CS_ID in X_code_snippet_ids]
#    print(X)
    X = X.to_numpy()
#    print(X[:20])
    seq_len = 128
#    print(X.shape)
    if args.mode == "code":
        X = X.reshape(X.shape[0], seq_len, 1)
        X = np.repeat(X, 2, axis=2)
    else:
        X = X.reshape(X.shape[0], seq_len, 769)


    # reproducibility
 # fix random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Ensures deterministic behavior for CuDNN
    # Set Python seeds
    random.seed(seed)
    np.random.seed(seed)
    # Set TensorFlow seeds
    tf.random.set_seed(seed)
    # Configure TensorFlow to be deterministic
    tf.config.experimental.enable_op_determinism()
#    fold_name='_'.join(test_stim)
    # define k-fold cross validation
    cvscores = []
    aucscores = []
    f1scores=[]
    recallscores=[]
    results_dir='./result_stim'
    learning_r=0
    # train and evaluate neural network
    os.makedirs(
        f'{results_dir}/cc-{args.simulation}-{args.problem_setting}-{args.mode}{args.less_fold}',
        exist_ok=True,
    )
#    i=0
    for fold, (train, test) in enumerate(train_test_splits):

        if fold < args.fold_offset:
            print(f"Skipping Fold {fold}")
            continue
        # clear tensorflow keras global state memory to start with blank state at each iteration
        keras.backend.clear_session()

        # scale fixation duration
        scaler = StandardScaler()
#        X[train][:, 1] = scaler.fit_transform(X[train][:, 1])
#        X[test][:, 1] = scaler.transform(X[test][:, 1])
#        print(X[train][:, :, 0])
#        print(X[train][:, :, 0].dtype)
        X_code_snippet_IDXs = np.array(X_code_snippet_IDXs)

        # define model
        def build_model(hp):
            print(f'{fold=}')
            # position embedding input
            code_snippet_idxs = Input(shape=(1,))
            # IA_ID sequence
            IA_ID_sequence = pos_emb_input = Input(shape=(seq_len,))
#            print("pos:")
#            print(pos_emb_input)
            # fixation duration input
#            fix_dur_input = Input(shape=(seq_len, 1))

            # decide whether to use graphbert or bert
            _code_lm_model = hp.Choice(
                'code-lm-embedding',
                values=['codebert'],
            )

            attention_window_size = hp.Choice(
                'attention-window-size',
 #               values=[0, 1, 2], # corresponds to total window sizes [1, 3, 5]
                values=[1],
            )
            if args.mode=='code':
                code_context_input = AttentionLayer(
                    data=np.load(f'../processed_data/PRE_COMPUTED_EMBEDDINGS_CODE.npy'),
                    window_size=attention_window_size,
                    lm=_code_lm_model,
                )((IA_ID_sequence, code_snippet_idxs))
            else:
                if args.simulation=='-7':
                    data = np.load(
                    '../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_float16.npy')
                elif args.simulation=='REA':
                    data=np.load(
                    '../processed_data/PRE_COMPUTED_EMBEDDINGS_CODE.npy')
                elif args.simulation=='zero':
                    data=np.load(
                    '../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_ZERO.npy')
                elif args.simulation=='random':
                    data=np.load(
                    '../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_RANDOM.npy')
                elif args.simulation=='randomall':
                    data=np.load(
                    '../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_RANDOM_OVERALL.npy')
                elif args.simulation=='-9':
                    data=np.load(
                    '../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_-9.npy')
                elif args.simulation=='real':
                    data=np.load(
                    '../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_REAL.npy')
#                else:
#                    data=np.load(f'../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_REAL_PLUS{args.simulation[4:]}.npy')
                elif args.simulation in [str(i) for i in range(0,11)]:
                    data=np.load(
                    f'../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_ZERO.npy')
                elif args.simulation in ['REA'+str(i) for i in range(0,11)]:
                    data=np.load(
#                    f'../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_REA_{args.simulation[3:]}.npy')
                    f'../processed_data/PRE_COMPUTED_EMBEDDINGS_CODE.npy')
                elif int(args.simulation)>20 :
                    data=np.load(
                    f'../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_ZERO.npy')

                code_context_input = AttentionLayer(
                    data=data,
                    window_size=attention_window_size,
                    lm=_code_lm_model,
                )((IA_ID_sequence, code_snippet_idxs))

            # decide which layer type to use (BiLSTM vs LSTM)
            _seq_model_type = hp.Choice(
                'layer_type',
#                values=['bidirectional', 'unidirectional'] # values=['bidirectional', 'unidirectional', 'self_attention']
                 values=['bidirectional']
            )

            no_lstm_layers = 2
            lstm_units = hp.Choice('lstm_units', values=[32])
            num_heads = 0 # num_heads = hp.Choice('num_heads', values=[2, 4, 8, 16])
            # decide whether to use bottleneck layer for embeddings
            _use_embedding_bottleneck = hp.Choice(
                'use_embedding_bottleneck',
                values=[True],
            )
            if _use_embedding_bottleneck:
                code_context_input = Dense(2*lstm_units)(code_context_input)

            # decide whether to use positional embedding (from token sequence)
            _positional_embedding = hp.Choice(
                'use_positional_embedding',
                values=[True],
            )
            if args.mode == 'bimodal':
                # instanitate positional embedding
                if _positional_embedding:
                    if _use_embedding_bottleneck:
                        pos_embedding = Embedding(512, 2*lstm_units)(pos_emb_input) # project index to 512 dim space
                        inputs = Add()([pos_embedding, code_context_input]) # add pos_embedding code_context
                    else:
                        pos_embedding = Embedding(512, 768)(pos_emb_input)
                        inputs = Add()([pos_embedding, code_context_input])
                else:
                    inputs = code_context_input
            elif args.mode == 'code':
                # just use code embedding without fixations
                inputs = code_context_input
            # if mode bimodal/fixations, add fixation durations
            if args.mode == 'bimodal' or args.mode == 'fixations':
                inputs = Concatenate()([inputs])
            if _seq_model_type == "self_attention":
                _pooling_type = 'max'
            else:
                _pooling_type = hp.Choice(
                    'pooling_type',
#                    values=['last_hidden_state', 'avg', 'max'],
                    values=['last_hidden_state'],
                )
            for no_lstm_layer in range(no_lstm_layers):
                if no_lstm_layer == 0:
                    if _seq_model_type == 'bidirectional':
                        lstm_output = Bidirectional(
                            LSTM(lstm_units, return_sequences=True),
                        )(inputs)
                    elif _seq_model_type == 'unidirectional':
                        lstm_output = LSTM(
                            lstm_units, return_sequences=True,
                        )(inputs)
                    else:
                        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)
                        lstm_output = mha(inputs, inputs)  # self-attention
                elif no_lstm_layer == (no_lstm_layers - 1):
                    if _seq_model_type == 'bidirectional':
                        if _pooling_type == 'last_hidden_state':
                            lstm_output = Bidirectional(
                                LSTM(lstm_units),
                            )(lstm_output)
                        elif _pooling_type == 'max':
                            lstm_output = Bidirectional(
                                LSTM(lstm_units, return_sequences=True),
                            )(lstm_output)
                            lstm_output = GlobalMaxPooling1D()(lstm_output)
                    elif _seq_model_type == 'unidirectional':
                        if _pooling_type == 'last_hidden_state':
                            lstm_output = LSTM(lstm_units)(lstm_output)
                        elif _pooling_type == 'max':
                            lstm_output = LSTM(lstm_units, return_sequences=True)(lstm_output)
                            lstm_output = GlobalMaxPooling1D()(lstm_output)
                        elif _pooling_type == 'avg':
                            lstm_output = LSTM(lstm_units, return_sequences=True)(lstm_output)
                            lstm_output = GlobalAveragePooling1D()(lstm_output)
                    else:
                        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)
                        mha_output = mha(inputs, inputs)  # self-attention

                            # as there is no last_hidden_state in MHA, we default to Max Pooling
                        if _pooling_type == 'max' or _pooling_type == 'last_hidden_state':
                            lstm_output = GlobalMaxPooling1D()(mha_output)
                        elif _pooling_type == 'avg':
                            lstm_output = GlobalAveragePooling1D()(mha_output)
                else:
                    if _seq_model_type == 'bidirectional':
                        lstm_output = Bidirectional(
                        LSTM(lstm_units, return_sequences=True),
                        )(lstm_output)
                    elif _seq_model_type == 'unidirectional':
                        lstm_output = LSTM(
                            lstm_units, return_sequences=True,
                        )(lstm_output)
                    else:
                        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)
                        lstm_output = mha(inputs, inputs)  # self-attention
            no_dense_layers=3
            dense_units = 64
            # instantiate dense layers
            if no_dense_layers == 1:
                dense_output = Dense(1, activation='sigmoid')(lstm_output)
            else:
                for no_dense_layer in range(no_dense_layers):
                    if no_dense_layer == 0:
                        dense_output = Dense(dense_units, activation='relu')(lstm_output)
                    elif no_dense_layer == (no_dense_layers - 1):
                        dense_output = Dense(1, activation='sigmoid')(dense_output)
                    else:
                        dense_output = Dense(dense_units, activation='relu')(dense_output)
            model = Model(
                inputs=[
                    code_snippet_idxs, pos_emb_input,
                ],
                outputs=dense_output,
            )
            if args.simulation=='real':
                learning_r=1e-5
                lr_schedule = ExponentialDecay(
                    initial_learning_rate=learning_r,
                    decay_steps=10000,
                    decay_rate=0.9,
                    staircase=True  # if True, decay the learning rate at discrete intervals
                )

                opt=Adam(learning_rate=lr_schedule)
            else:
                learning_r=5e-4
                opt = Adam(learning_rate=learning_r)
            
            if args.problem_setting == 'subjective_difficulty_score':
                # compile model with MSE loss and MSE & RMSE metrics
                 model.compile(
                    loss='mean_squared_error',
                    optimizer=opt,
                    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError(name='rmse')],
                )
            else:
                # compile model with BCE loss and accuracy & AUC metrics
                model.compile(
                    loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=[ 'accuracy',auroc,f1_metric,recall_metric,],
                )

            return model
        if args.problem_setting == "subjective_difficulty_score":
            objective = keras_tuner.Objective('val_loss', direction='min')
            epochs = 200
        else:
            objective = keras_tuner.Objective('val_loss', direction='min')
            epochs = 50
        directory=f'{results_dir}/cc-{args.simulation}-{args.problem_setting}-{args.mode}{args.less_fold}'
        tuner = keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective=objective,
            directory= directory,
            max_trials=1,
            project_name=f'{fold}',
            overwrite=True,
        )
        csv_logger = CSVLogger(f'{directory}/{fold}/training_log.csv', append=False)
        # implement callbacks
        callbacks = [
            csv_logger,
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
            ),
        ]
#        print(y)
        # tune hyperparameters
        tuner.search(
            x=[X_code_snippet_IDXs[train],
                X[train][:, :, 0].astype(np.float64),
            ],
            y=y[train],
            batch_size=16,
            validation_split=0.1,
            epochs=epochs, # 1000 or 200
            callbacks=callbacks,
            verbose=1,
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        model = tuner.hypermodel.build(best_hp)
        model.fit(
            [
                X_code_snippet_IDXs[train],
                X[train][:, :, 0].astype(np.float64),
            ],
            y[train].astype(np.float64),
            epochs=epochs, # 1000 or 200
            batch_size=16,
            verbose=1,
            validation_split=0.1,
            callbacks=callbacks,
        )
#        print(y[test])
#        print(y[])
        # calculate scores
#        print(learning_rate)
        scores = model.evaluate(
            [
                # CodeSnippet ID
                X_code_snippet_IDXs[test],
                # IA ID
                X[test][:, :, 0].astype(np.float64),
#                print(X[test][:,:,0])
                # FIX DUR
#                X[test][:, :, 1],
            ],
            y[test],
            verbose=1,
        )
        df_result=pd.read_csv(args.output)
        print(type(fold),fold)
        # if fold==2:
        #     cmap = LinearSegmentedColormap.from_list("custom_red", ["white", "red"])
        #
        #     for layer in model.layers:
        #         weights = layer.get_weights()  # Returns a list of numpy arrays
        #         if weights:  # Check if the layer has weights
        #             weight_matrix = weights[0]  # Assuming the first element is the weight matrix
        #             if weight_matrix.ndim == 2:  # Ensure the weights are 2D
        #                 vmin = -0.3  # Set the minimum value of the color scale
        #                 vmax = 0.3  # Set the maximum value of the color scale
        #
        #                 plt.figure(figsize=(12, 8))
        #                 sns.heatmap(weight_matrix, annot=False, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax)
        #                 plt.title(f"Heatmap of Weights in Layer: {layer.name}")
        #                 name = f'{args.simulation}-{layer.name}-{args.mode}{str(fold)}'
        #                 plt.savefig(f"../figure/{name}.png")
        #                 plt.close()  # Ensure the plot is closed after saving
        #             else:
        #                 print(f"Skipping {layer.name}, weight matrix not 2D.")
        #         else:
        #             print(f"No weights to plot for layer: {layer.name}")

        print(scores)
        if args.problem_setting != "subjective_difficulty_score":
            print(
                f'{model.metrics_names[1]} {scores[1]*100:.2f}\t'
                f'{model.metrics_names[2]} {scores[2]*100}\t',
            )
            # save scores
            cvscores.append(scores[1] * 100)
            aucscores.append(scores[2] * 100)
            f1scores.append(scores[3]*100)
            recallscores.append(scores[4]*100)
            accuracy=round(np.mean(cvscores),2)
            accuracy_var=round(np.std(cvscores) / np.sqrt(len(cvscores)),2)
            auc=round(np.mean(aucscores),2)
            auc_var=round(np.std(aucscores) / np.sqrt(len(aucscores)),2)
            recall=round(np.mean(recallscores),2)
            recall_var=round(np.std(recallscores) / np.sqrt(len(recallscores)),2)
            f1=round(np.mean(f1scores),2)
            f1_var=round(np.std(f1scores) / np.sqrt(len(f1scores)),2)
            print(cvscores,aucscores,f1scores)
            print(
                f'accuracy: {accuracy}% '
                f'(+/- {accuracy_var}%)',
            )         
            print(
                f'AUC: {auc}% '
                f'(+/- {auc_var}%)',
            )
            print(
                f'F1: {f1}% '
                f'(+/- {f1_var}%)',
            )
            print(
                f'Recall: {recall}% '
                f'(+/- {recall_var}%)',
            )
            print(learning_r)
            df_result.loc[len(df_result)]=[f'{args.simulation}-{args.problem_setting}-{args.mode}{args.less_fold}',accuracy,accuracy_var,recall,recall_var,f1,f1_var,auc,auc_var,args.seed]
            df_result.to_csv(args.output,index=False)
        else:
            print(scores)
            print(
                f'{model.metrics_names[1]} {scores[1] * 100:.2f}\t'
                f'{model.metrics_names[2]} {scores[2] * 100}\t'
            )
            # save scores
            cvscores.append(scores[1] * 100)
            aucscores.append(scores[2] * 100)

            print(
                f'MAE: {np.mean(cvscores):.2f}% '
                f'(+/- {np.std(cvscores) / np.sqrt(len(cvscores)):.2f}%)',
            )
            print(
                f'RMSE: {np.mean(aucscores):.2f}% '
                f'(+/- {np.std(aucscores) / np.sqrt(len(aucscores)):.2f}%)',
            )

        # Fold Summary
        model.summary()
        total_trainable_params = model.count_params()
        print("Total trainable parameters:", total_trainable_params)
    # Evaluation Summary
    model.summary()
    total_trainable_params = model.count_params()
    print("Total trainable parameters:", total_trainable_params)
    end_time = time.time()
    print('time trained:', end_time - start_time)

    return 0


if __name__ == '__main__':
#    test_stim = [
#        ['stim1', 'stim16','stim7'],
#        ['stim2', 'stim22','stim13'],
#        ['stim15', 'stim19','stim5'],
#        ['stim24','stim21']
#    ]
    test_stim = [
    ['stim1', 'stim16', 'stim7', 'stim24'],
    ['stim2', 'stim22', 'stim13', 'stim21'],
    ['stim15', 'stim19', 'stim5'],
]
#    test_stim = [
#        ['stim1', 'stim16','stim7','stim24','stim21','stim15'],
#        ['stim2', 'stim22','stim13', 'stim19','stim5'],
#    ]
#    stimlist=['stim1','stim2','stim15','stim16','stim19','stim22']
    X = load_data_stim(args.mode,args.simulation)
#    print(X)
    main(test_stim, X)
#    print(test_stim)
"""
        # Generate all combinations of 2 elements
    combinations = list(itertools.combinations(stimlist, 2))
        # Shuffle the list of combinations
    random.shuffle(combinations)
        # Return or process each combination
    for combination in combinations:
        test_stim = [combination[0],combination[1]]
        X_copy = copy.deepcopy(X)  # Create a deep copy of X

        test_stim=[]
"""

