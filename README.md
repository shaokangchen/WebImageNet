# WebImageNet
classify images from image net with web API
using BVLC alexnet caffee model.

client need to load a file contains images links at each line.
client encode the link files as queries by json string and send to server.
server decode the json string and run the classify for each query image.
server then send the result as json string to client.

users can follow the imagelist.txt file to create their own list.
The object classification is using BVLC alexnet caffee model from 
https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet. 

