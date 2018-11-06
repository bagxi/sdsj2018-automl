import argparse
import os
import time
import pickle
from main import automl

TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression','predict'], required=True)
    args = parser.parse_args()    
    
    if args.mode == 'predict':
        aml_filename = os.path.join(args.model, 'aml.pkl')
        with open(aml_filename, 'rb') as f:
            aml = pickle.load(f)
            aml.tl = TIME_LIMIT
        df = aml.predict(args.input)
        df[['line_id', 'prediction']].to_csv(args.output, index=False)
    else:
        aml = automl(args.mode,TIME_LIMIT)
        aml.train(args.input)
        aml_filename = os.path.join(args.model, 'aml.pkl')        
        with open(aml_filename, 'wb') as f_stream:
            pickle.dump(aml, f_stream, protocol=pickle.HIGHEST_PROTOCOL)