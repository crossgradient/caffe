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
from numpy import shape, mean


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
        in_df = pd.io.parsers.read_csv(args.input_file,sep='\t',header=None,names=['file'])
    elif os.path.isdir(args.input_file):
        filenames = glob.glob(args.input_file + '/*.' + args.ext)
        inputs =[caffe.io.load_image(im_f,color=False)
                 for im_f in filenames]
    else:
        inputs = [caffe.io.load_image(args.input_file,color=False)]

    labels = np.load('labels.npy').astype('string')

    labelsLookup = {}
    i = 0
    print labels
    for l in labels:
        labelsLookup[l] = i
        i = i+1

    def makePng(x):
        name_parts = rsplit(x,'.',1)
        return name_parts[0] + '.jpg'

    in_df['png'] = in_df['file'].apply(makePng)

    w = None 
    f = []
    in_df['fullfile'] = in_df['file'].apply(lambda(x): '../../data/12130/test2/' + x) 
    batchSize = 13040
    batches = len(in_df) / batchSize
    for i in range(0,batches) :
        start = time.time()
        currentBatch = in_df[(in_df.index >= i*batchSize) & (in_df.index < (i+1)*batchSize)]
        inputs = [caffe.io.load_image(im_f,color=False) for im_f in currentBatch.fullfile.values]
        predictions = classifier.predict(inputs, False, False)
	#print predictions
        if w is None :
		w = predictions
	else : 
		w = np.append(w,predictions,axis=0)
        #print w
	f.append(currentBatch.png.values)
        print "Done in %.2f s." % (time.time() - start)
    #w_ = np.array(w).flatten()
    fileNames_df = pd.DataFrame(np.array(f).flatten())
    #print fileNames_df
    preds_df = pd.DataFrame(w,columns=labels)
    #print shape(preds_df)
    #cleanNames = pd.DataFrame(fileNames_df[0].apply(doTrim))
    foo = fileNames_df.join(preds_df)
    foo.rename(columns={0:'image'}, inplace=True)
    colz = np.concatenate([['image'],labels])

    #preds_df = pd.DataFrame(np.transpose(np.array([f_,w_])),columns=['file','label']) 
    #preds_df['fullfile'] = preds_df['file'].apply(lambda(x): '../../data/12130/test/' + x)
    foo.to_csv(args.output_file,index=False,header=True, sep=',',columns=colz)

if __name__ == '__main__':
    main(sys.argv)
