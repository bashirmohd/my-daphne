from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml
import pandas as pd
import glob

from lib.utils import load_graph_data
from lib.utils import generate_seq2seq_data
from lib.utils import train_val_test_split
from lib.utils import StandardScaler
from lib.utils import MinMaxScaler

from model.dcrnn_supervisor import DCRNNSupervisor

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        supervisor_config['model']['num_nodes'] = num_nodes = len(sensor_ids)
        
        batch_size = supervisor_config.get('data').get('batch_size')
        val_batch_size = supervisor_config.get('data').get('val_batch_size')
        test_batch_size = supervisor_config.get('data').get('test_batch_size')
        horizon = supervisor_config.get('model').get('horizon')
        seq_len = supervisor_config.get('model').get('seq_len')

        # Data preprocessing 
        validation_ratio = supervisor_config.get('data').get('validation_ratio')
        test_ratio = supervisor_config.get('data').get('test_ratio')

        #filename = supervisor_config['data']['dataset_dir'] + 'ts_0.h5'
        #print('Filename',filename)
        #data = pd.read_hdf(filename)
        #train_scale, _, _ = train_val_test_split(data, val_ratio=validation_ratio, test_ratio=test_ratio)
        #scaler_list = StandardScaler(mean=train_scale.mean(), std=train_scale.std())
        #print(train_scale.mean(), train_scale.std())
        
        data_tr = []
        data_v = []
        data_te = []
        scaler_list = []
        for i in range(0,7):
            #filename = supervisor_config['data']['dataset_dir'] + 'speed.h5'
            filename = supervisor_config['data']['dataset_dir'] + 'ts_' + str(i) + '.h5'
            #filename = supervisor_config['data']['dataset_dir'] + 'ts_' + str(i) + str(i) + '_log.h5'

            data = pd.read_hdf(filename)
            train, val, test = train_val_test_split(data, val_ratio=validation_ratio, test_ratio=test_ratio)
            #scaler = StandardScaler(mean=train.mean(), std=train.std())
            scaler = MinMaxScaler(min_val=train.min(), max_val=train.max())
            scaler_list.append(scaler)
            data_tr.append(train)
            data_v.append(val)
            data_te.append(test)
        

        data_train = generate_seq2seq_data(data_tr, batch_size, seq_len, horizon, num_nodes, 'train', scaler_list)
        data_val = generate_seq2seq_data(data_v, val_batch_size, seq_len, horizon, num_nodes, 'val', scaler_list)
        data_train.update(data_val)

        # data_test = generate_seq2seq_data(data_te, test_batch_size, seq_len, horizon, num_nodes, 'test', scaler_list)
        data_test = generate_seq2seq_data(data_te, test_batch_size, seq_len, horizon, num_nodes, 'test', scaler_list)
  
        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx, data_train, supervisor_config)
            # Train
            data_tag = supervisor_config.get('data').get('dataset_dir')
            folder = data_tag + '/model/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            supervisor.train(sess=sess)
            

            # Test
            yaml_files = glob.glob('%s/model/*/*.yaml'%data_tag, recursive=True)
            yaml_files.sort(key=os.path.getmtime)
            config_filename = yaml_files[-1] #'config_%d.yaml' % config_id
            
            with open(config_filename) as f:
                config = yaml.load(f)
            # Load model and evaluate
            supervisor.load(sess, config['train']['model_filename'])

            y_preds = supervisor.evaluate(sess, data_test)
            
            n_test_samples = data_test['y_test'].shape[0]
            print('n_test_samples', n_test_samples)
            folder = data_tag + '/results/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            '''
            for feature_i in range(0,2):
                data = pd.read_hdf(supervisor_config['data']['dataset_dir'] + 'ts_' + str(feature_i) + '.h5')
                #tr, _, eval_dfs = train_val_test_split(data, val_ratio=validation_ratio, test_ratio=test_ratio)
                y_pred = pd.DataFrame(y_preds[:, 0, :, feature_i], columns=data.columns) 
                #print('y_pred', y_pred.columns)
                filename = os.path.join('%s/results/'%data_tag, 'dcrnn_prediction_%d.h5' %feature_i)
                y_pred.to_hdf(filename, 'results')


            '''
            for horizon_i in range(24):
                for feature_i in range(0,7):
                    filename = supervisor_config['data']['dataset_dir'] + 'speed.h5'
                    filename = supervisor_config['data']['dataset_dir'] + 'ts_' + str(feature_i) + '.h5'
                    #filename = supervisor_config['data']['dataset_dir'] + 'ts_' + str(feature_i) + str(feature_i) + '_log.h5'

                    data = pd.read_hdf(filename)
                    tr, _, test = train_val_test_split(data, val_ratio=validation_ratio, test_ratio=test_ratio)
                    
                    eval_dfs = test[seq_len + horizon_i: seq_len + horizon_i + n_test_samples]
                    #eval_dfs = scaler_list[feature_i].transform(eval_dfs)
                    #eval_dfs = tr[seq_len + horizon_i: seq_len + horizon_i + n_test_samples]

                    df_sp = pd.DataFrame(y_preds[:, horizon_i, :, feature_i], index=eval_dfs.index, columns=eval_dfs.columns) 
                    y_pred = df_sp
                    y_pred = scaler_list[feature_i].inverse_transform(df_sp)

                    filename = os.path.join('%s/results/'%data_tag, 'dcrnn_prediction_%s_h%s.h5' %(str(feature_i),str(horizon_i)))
                    y_pred.to_hdf(filename, 'results')

                    filename = os.path.join('%s/results/'%data_tag, 'ts_test_%s_h%s.h5' %(str(feature_i),str(horizon_i)))
                    eval_dfs.to_hdf(filename, 'results')
            
                
            print('Predictions saved as %s/results/dcrnn_prediction_[1-12].h5...' %data_tag)
            
          


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
