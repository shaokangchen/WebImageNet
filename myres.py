import json
import os
import sys
import caffe
import urlparse, mimetypes
import requests
from StringIO import StringIO
import numpy
import cv2
import re
import copy
from pathos.multiprocessing import ProcessingPool as Pool
#from multiprocessing import Pool

### check if url is valid for image format ###
def is_url_image(url):
        maintype= mimetypes.guess_type(urlparse.urlparse(url).path)[0]
        return (maintype and maintype.startswith('image'))

### load imagenet labels for the model    
def load_labels(dictfile):
    class_labels = []
    try:
        if(os.path.exists(dictfile)):
            fp = open(dictfile,'rt')
            lines = fp.readlines()
            fp.close()
            for line in lines:                
                splits = re.split("[:', ]", line)
                class_labels.append(splits[4])
        
    except:
        print( "error reading label file" )
        raise IOError("cannot read label file " + dictfile)
    return class_labels

### define a single classify result for one class label
class result:    
    labels = load_labels('./imagenet/class_labels.txt')
    def __init__(self):
        self.name = '';
        self.confidence = 0.0;
        self.location = [0,0,0,0]

    def set_result(self, name, conf):
        self.name = name
        self.confidence = conf

    def set_labels(self, class_idx, conf):
        self.name = result.labels[class_idx]
        self.confidence = conf

    def set_location(self, loc):
        self.location = loc

    def toJSON(self):
        mydict = {'class' : self.name, 'confidence':self.confidence, 'location':self.location}
        #return dict(classes=self.name, confidence=self.confidence)
        return mydict
    
### define  a list of results for multiple class labesl
class myres:
    def __init__(self):
        self.results = [ ]

    def add(self, result):
        self.results.append(result)

    def toJSON(self):
        return self.results

### define a query for client
class myquery:
    def __init__(self, filename=None):
        self.url_list = []
        self.count = 0
        self.threshold = 0.4
        if(filename != None):
            self.getlist_fromfile(filename)

    def set_threshold(self, threshold):
        self.threshold =  threshold

### load all urls from a file ###
    def getlist_fromfile(self,filename):
        self.url_list = []
        self.count = 0
        try:
            if(os.path.exists(filename)):
                fp = open(filename,'rt')
                lines = fp.readlines()
                fp.close()
                for line in lines:
                    newline = line.strip()                    
                    #id = uuid.uuid4()
                    self.url_list.append(newline)                        
                    self.count += 1
                    #print(id) 
            else:
                print ("file " + filename + " does not exists")
            print('totally ' , self.count , ' URL links collected ')
        except:
            print ("error reading files")
            raise IOError("cannot read url list " + filename)
    def toJSON(self):
        return dict(images=self.url_list,threshold=self.threshold)

### define a decode of query from JSON string    
    @classmethod
    def fromJSON(cls, json_string):
        queries = json.loads(json_string)
        cls.url_list = queries['images']
        cls.count = len(cls.url_list)
        cls.threshold = queries['threshold']
        return cls

### define group of results for a single URL query 
class group_results:
    #caffe.set_device(0)
    #caffe.set_mode_gpu()
    try:

        net = caffe.Net('./imagenet/deploy.prototxt', './imagenet/bvlc_alexnet.caffemodel', caffe.TEST)
    except:
        print('cannot load model file')
        raise IOError("cannot load model files.")
    def __init__(self, threshold):
        self.url=''
        self.results=[]
        self.valid = False
        self.conf_threshold = threshold
    def set_url(self, url):
        self.url=url
        self.results=[]
        self.valid = False
    def is_url_image(self,url):
        try:
            maintype= mimetypes.guess_type(urlparse.urlparse(url).path)[0]
        except:
            print('cannot check if valid url or not')
            return False
        return (maintype and maintype.startswith('image'))

    def download_img(self, url):    
        try:    
            data = requests.get(url)
        except Exception,error:
            print(error)
            return None
        ### check response status
        if(data.status_code > 300):
            print('error')
            return None
        ### convert the download data to opencv image
        file = StringIO(data.content)
        file = numpy.asarray(bytearray(data.content), dtype="uint8")
        img = cv2.imdecode(file, cv2.IMREAD_COLOR)
        return img
    def toJSON(self):
        if self.valid:
            return dict(url=self.url, classes=self.results)
        else:
            return dict(url=self.url, error=self.results)

    def classify(self,image):
        img = cv2.resize(image, (227,227), interpolation=cv2.INTER_CUBIC)
        #avg = numpy.array([93.5940,104.7624,129.1863]) # BGR mean from VGG
        avg = numpy.array([104.0069879317889,116.66876761696767,122.6789143406786]) # mean from imagenet
        img = img-avg # subtract mean
        img = img.transpose((2,0,1)) # to match image input dimension: 3x227x227
        img = img[None,:] # add singleton dimension to match batch dimension        
        res = group_results.net.forward_all(data=img)
        return res['prob'][0]

### sliding window scan of an image
    def scan_image(self, image, win_size, step):
        crop_list = []
        crop_loc = []
        row, col,channel = image.shape
        for i in range(0, row - win_size, step):
            for j in range(0, col - win_size, step):
                cropped = image[i:i+win_size,j:j+win_size,:]
                crop_list.append(cropped)
                crop_loc.append([i,j,i+win_size,j+win_size])
        return crop_list, crop_loc

### classify each sliding window
    def search_image(self, image_list):
        res_list = []
        for image in image_list:            
            tmp_res = self.classify(image)
            res_list.append(tmp_res)
        return res_list

### check the overlap of two sliding windows   
    def window_dist(self,window1, window2):
        min_i = min(window1[0],window2[0])
        min_j = min(window1[1],window2[1])
        max_i = max(window1[2],window2[2])
        max_j = max(window1[3],window2[3])
        mat1 = numpy.zeros([max_i,max_j])
        mat2 = numpy.zeros([max_i,max_j])
        mat1[window1[0]:window1[2],window1[1]:window1[3]] = 1
        mat2[window2[0]:window2[2],window2[1]:window2[3]] = 1
        overlap = numpy.logical_and(mat1,mat2)
        combine = numpy.logical_or(mat1,mat2)
        ratio = float(numpy.sum(overlap)) / numpy.sum(combine)
        return ratio

### merege the classify results of closing windows    
    def merge_res(self, res_list, location_list):
        count = numpy.zeros([len(res_list)])
        res_sum = copy.deepcopy(res_list)
        ### sum the results of closing windows with corresponding number of count
        for i in range(0, len(res_list)):
            for j in range(i+1, len(res_list)):
                if(self.window_dist(location_list[i],location_list[j])>0.3):
                    res_sum[i] = res_sum[i] + res_list[j]
                    res_sum[j] = res_sum[j] + res_list[i]
                    count[i] += 1
                    count[j] += 1
        ### apply average to each individual sliding window
        for i in range(0,len(res_sum)):
            res_sum[i] = res_sum[i] / count[i]
        return res_sum

### get the maximum value for each class label from all the results
### TODO: should return the best location for each class label
    def find_result(self, res_list, location_list):
        res_array = numpy.asarray(res_list)
        max_column = numpy.argmax(res_array, axis=0)
        max_res = res_array[max_column,range(0,len(res_list[0]))]
        #sort_row = numpy.argsort(max_res)
        return max_res, max_column

### exahustive classify of an image based on sliding windows
##### TODO:  allow users to set the scan window size and steps###
### TODO:  apply image segmentation to generate windows
    def classify_exhaust(self,image):
        win_size = image.shape[0]*2/3
        crop_list, crop_loc = self.scan_image(image, win_size, win_size/5)
        win_size = image.shape[0]*3/4
        crop_list2, crop_loc2 = self.scan_image(image, win_size, win_size/5)
        #win_size = image.shape[0]*2/3
        #crop_list3, crop_loc3 = self.scan_image(image, win_size, win_size/5)
        crop_list += crop_list2 #+ crop_list3
        crop_loc +=  crop_loc2 #+ crop_loc3
        sub_results = self.search_image(crop_list)
        results_merge = self.merge_res(sub_results, crop_loc)
        results_max, results_idx = self.find_result(results_merge, crop_loc)        
        results_loc = [crop_loc[i] for i in results_idx]
        return results_max, results_loc

### run a exhaustive classify on a query url by scanning windows
### return the location of the class labels to users
    def run_classify_exhaust(self):
        if( self.is_url_image(self.url) ):
            image = self.download_img(self.url)
            ### if invalid image download, return the error message
            if(image is None):
                self.valid = False
                self.results = "cannot download image."
            else:
                classify_res, classify_loc = self.classify_exhaust(image)
                ### sort the results based on probability
                sort_idx = sorted(range(len(classify_res)), key=lambda k: classify_res[k], reverse=True)
                res = myres()
                for i in range(0,len(sort_idx)):
                    ### if probability is larger then the threshold add it to the result
                    if(classify_res[sort_idx[i]]>self.conf_threshold):
                        single_res = result()
                        #single_res.set_labels(sort_idx[i],classify_res[sort_idx[i]])
                        single_res.set_labels(sort_idx[i],round(classify_res[sort_idx[i]], 3))
                        single_res.set_location(classify_loc[sort_idx[i]])
                        #print json.dumps(single_res, cls=ComplexEncoder)
                        res.add(single_res)
                        #crop_img = image[classify_loc[sort_idx[i]][0]:classify_loc[sort_idx[i]][2],classify_loc[sort_idx[i]][1]:classify_loc[sort_idx[i]][3],:]
                        #cv2.imshow(single_res.name, crop_img)
                        #cv2.waitKey(0)
                    else:
                        break
                #print json.dumps(res, cls = ComplexEncoder)
                ### if no class label in results is high enough, then label it as "unknown"
                if(len(res.results)>0):
                    self.results = res         
                    self.valid = True       
                else:
                    self.results = "unknown"
                    self.valid = True  
        else:
            self.valid = False
            self.results = "invalid URL"

### run a simple classify of a query url with only one window
### TODO: allow users to choose query mode: simple or exhaustive   
    def run_classify(self):
        if( self.is_url_image(self.url) ):
            image = self.download_img(self.url)
            if(image is None):
                self.valid = False
                self.results = "cannot download image."
            else:
                classify_res = self.classify(image)
                sort_idx = sorted(range(len(classify_res)), key=lambda k: classify_res[k], reverse=True)
                res = myres()
                for i in range(0,len(sort_idx)):
                    if(classify_res[sort_idx[i]]>self.conf_threshold):
                        single_res = result()
                        #single_res.set_labels(sort_idx[i],classify_res[sort_idx[i]])
                        single_res.set_labels(sort_idx[i],round(classify_res[sort_idx[i]], 3))
                        #print json.dumps(single_res, cls=ComplexEncoder)
                        res.add(single_res)
                    else:
                        break
                #print json.dumps(res, cls = ComplexEncoder)
                if(len(res.results)>0):
                    self.results = res         
                    self.valid = True       
                else:
                    self.results = "unknown"
                    self.valid = True  
        else:
            self.valid = False
            self.results = "invalid URL"

### define a list of results for multiple url queries ###
class result_list:
    def __init__(self):
        self.results = []
    def add(self,result):
        self.results.append(result)
    def toJSON(self):
        return dict(results=self.results)

    def single_query(self, url, threshold):
        g = group_results(threshold)
        g.set_url(url)
        g.run_classify_exhaust()        
        return g    
    def run_queries(self, query):  
        ### run a list of queries in multiple processes      
        p = Pool(2)
        thresholds = [ query.threshold] *  len(query.url_list)
        results = p.amap(self.single_query, query.url_list, thresholds)
        for res in results.get():
            self.add(res)

        ### run each url query in order
        #for q in query.url_list:
        #    g = group_results(query.threshold)
        #    g.set_url(q)
        #    #g.run_classify()
        #    g.run_classify_exhaust()
        #    #print(json.dumps(g,cls=ComplexEncoder, indent=4))
        #    self.add(g)
    def conver_json(self):
        return json.dumps(self.toJSON(),cls=ComplexEncoder, indent=4)

### an encoder for JSON, each class should define .toJSON()
class ComplexEncoder(json.JSONEncoder):
    def default(self,obj):
        if hasattr(obj,'toJSON'):
            return obj.toJSON()
        else:
            return json.JSONEncoder.default(self,obj)
