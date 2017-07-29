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
from multiprocessing import Pool


def is_url_image(url):
        maintype= mimetypes.guess_type(urlparse.urlparse(url).path)[0]
        return (maintype and maintype.startswith('image'))

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
        
    except ValueError:
        print( "error reading label file" )
    return class_labels

class result:
    #labels = ['person', 'dog', 'car', 'tree', 'water']
    labels = load_labels('./imagenet/class_labels.txt')
    def __init__(self):
        self.name = '';
        self.confidence = 0.0;

    def set_result(self, name, conf):
        self.name = name
        self.confidence = conf

    def set_labels(self, class_idx, conf):
        self.name = result.labels[class_idx]
        self.confidence = conf

    def toJSON(self):
        mydict = {'class' : self.name, 'confidence':self.confidence}
        #return dict(classes=self.name, confidence=self.confidence)
        return mydict
    

class myres:
    def __init__(self):
        self.results = [ ]

    def add(self, result):
        self.results.append(result)

    def toJSON(self):
        return self.results

class myquery:
    def __init__(self, filename=None):
        self.url_list = []
        self.count = 0
        if(filename != None):
            self.getlist_fromfile(filename)

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
                    print(id) 
            else:
                print ("file " + filename + " does not exists")
            print('totally ' , self.count , ' valid URL images collected ')
        except ValueError:
            print ("error reading files")
    def toJSON(self):
        return dict(images=self.url_list)

    
    @classmethod
    def fromJSON(cls, json_string):
        queries = json.loads(json_string)
        cls.url_list = queries['images']
        cls.count = len(cls.url_list)
        return cls


class group_results:
    #caffe.set_device(0)
    #caffe.set_mode_gpu()
    net = caffe.Net('./imagenet/deploy.prototxt', './imagenet/bvlc_alexnet.caffemodel', caffe.TEST)
    conf_threshold = 0.1
    def __init__(self):
        self.url=''
        self.results=[]
        self.valid = False
    def set_url(self, url):
        self.url=url
        self.results=[]
        self.valid = False
    def is_url_image(self,url):
        maintype= mimetypes.guess_type(urlparse.urlparse(url).path)[0]
        return (maintype and maintype.startswith('image'))
    def download_img(self, url):        
        data = requests.get(url)
        if(data.status_code > 300):
            print('error')
            return None
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
        avg = numpy.array([127,127,127]) # BGR mean from VGG
        img = img-avg # subtract mean
        img = img.transpose((2,0,1)) # to match image input dimension: 3x224x224
        img = img[None,:] # add singleton dimension to match batch dimension
        res = group_results.net.forward_all(data=img)
        return res['prob'][0]

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

    def search_image(self, image_list):
        res_list = []
        for image in image_list:            
            tmp_res = self.classify(image)
            res_list.append(tmp_res)
        return res_list

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

    def merge_res(self, res_list, location_list):
        count = numpy.zeros([len(res_list)])
        res_sum = copy.deepcopy(res_list)
        for i in range(0, len(res_list)):
            for j in range(i+1, len(res_list)):
                if(self.window_dist(location_list[i],location_list[j])>0.2):
                    res_sum[i] = res_sum[i] + res_list[j]
                    res_sum[j] = res_sum[j] + res_list[i]
                    count[i] += 1
                    count[j] += 1
        for i in range(0,len(res_sum)):
            res_sum[i] = res_sum[i] / count[i]
        return res_sum

    def find_result(self, res_list, location_list):
        res_array = numpy.asarray(res_list)
        max_column = numpy.argmax(res_array, axis=0)
        max_res = res_array[max_column,range(0,len(res_list[0]))]
        sort_row = numpy.argsort(max_res)

        return max_res


    def classify_exhaust(self,image):
        win_size = image.shape[0]*1/2
        crop_list, crop_loc = self.scan_image(image, win_size, win_size/4)
        win_size = image.shape[0]*2/3
        crop_list2, crop_loc2 = self.scan_image(image, win_size, win_size/4)
        win_size = image.shape[0]*3/4
        crop_list3, crop_loc3 = self.scan_image(image, win_size, win_size/4)
        crop_list += crop_list2 + crop_list3
        crop_loc +=  crop_loc2 + crop_loc3
        sub_results = self.search_image(crop_list)
        results_merge = self.merge_res(sub_results, crop_loc)
        results_max = self.find_result(results_merge, crop_loc)
        return results_max

    def run_classify_exhaust(self):
        if( self.is_url_image(self.url) ):
            image = self.download_img(self.url)
            if(image == None):
                self.valid = False
                self.results = "cannot download image."
            else:
                classify_res = self.classify_exhaust(image)
                sort_idx = sorted(range(len(classify_res)), key=lambda k: classify_res[k], reverse=True)
                res = myres()
                for i in range(0,len(sort_idx)):
                    if(classify_res[sort_idx[i]]>group_results.conf_threshold):
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
   
    def run_classify(self):
        if( self.is_url_image(self.url) ):
            image = self.download_img(self.url)
            if(image == None):
                self.valid = False
                self.results = "cannot download image."
            else:
                classify_res = self.classify(image)
                sort_idx = sorted(range(len(classify_res)), key=lambda k: classify_res[k], reverse=True)
                res = myres()
                for i in range(0,len(sort_idx)):
                    if(classify_res[sort_idx[i]]>group_results.conf_threshold):
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


class result_list:
    def __init__(self):
        self.results = []
    def add(self,result):
        self.results.append(result)
    def toJSON(self):
        return dict(results=self.results)
    def run_queries(self, query):
        for q in query.url_list:
            g = group_results()
            g.set_url(q)
            #g.run_classify()
            g.run_classify_exhaust()
            #print(json.dumps(g,cls=ComplexEncoder, indent=4))
            self.add(g)
    def conver_json(self):
        return json.dumps(self.toJSON(),cls=ComplexEncoder)

class ComplexEncoder(json.JSONEncoder):
    def default(self,obj):
        if hasattr(obj,'toJSON'):
            return obj.toJSON()
        else:
            return json.JSONEncoder.default(self,obj)
