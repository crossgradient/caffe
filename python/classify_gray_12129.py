#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob
import time
from string import rsplit

import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, gpu=args.gpu, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    if args.gpu:
        print 'GPU mode'

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        inputs = np.load(args.input_file)
    elif args.input_file.endswith('txt'):
        in_df = pd.io.parsers.read_csv(args.input_file,sep='\t',header=None,names=['id','file'])
    elif os.path.isdir(args.input_file):
        filenames = glob.glob(args.input_file + '/*.' + args.ext)
        inputs =[caffe.io.load_image(im_f,color=False)
                 for im_f in filenames]
    else:
        inputs = [caffe.io.load_image(args.input_file,color=False)]

    all_preds = None
    all_preds_names = None
    uniqueIds = in_df.id.unique()
    print "uniquesIds : " + str(len(uniqueIds))
    uniqueIdsSplit = np.split(uniqueIds,5)
    
    for currentId in range(0,5) :
        
        currentSplit = uniqueIdsSplit[currentId]
        currentBatch = in_df[in_df.id.isin(currentSplit)]
        fullPath = currentBatch['file'].apply(lambda(x):'/mnt/crossgradient/plankton/data/proctest/12129/'+x)
    
	print "Classifying batch " + str(currentId)

        inputs = [caffe.io.load_image(im_f,color=False)
                 for im_f in fullPath.values]
    	# Classify.
    	start = time.time()
    	predictions = classifier.predict(inputs, False, False) # not args.center_only)
    	print "Done in %.2f s." % (time.time() - start)

        # average them
	predictions_df = pd.DataFrame(predictions)
	newBatch = pd.concat([currentBatch.reset_index(),predictions_df],axis=1)
	batchAvg = newBatch.groupby(['id']).mean()
	batchAvg_arr = batchAvg.as_matrix()

	if all_preds == None :
		all_preds = batchAvg_arr
	else :
		all_preds = np.vstack((all_preds,batchAvg_arr))

        currentBatch['fileId'] = currentBatch['file'].apply(lambda(x):rsplit(x,'_',1)[1])
        batchNames = currentBatch.groupby('id').max()
	
	if all_preds_names == None :
		all_preds_names = batchNames.fileId.values
	else : 
		all_preds_names = np.hstack((all_preds_names,batchNames.fileId.values))
	
    np.save(args.output_file, all_preds)
    np.save(args.output_file + '_names', all_preds_names)

    #all_preds_df = pd.DataFrame(all_preds)
    #all_preds_df.to_csv('all_preds.csv', columns=['file','pred'], sep="\t", header=False, index=False)

if __name__ == '__main__':
    main(sys.argv)
