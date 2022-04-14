# Import essential libraries

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import urllib.request
import os
import glob
import csv
from PIL import ImageStat
from tqdm import tqdm
import colorgram
from cv2 import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import pickle
from multiprocessing import Pool, cpu_count

# Define class with GET_IMAGES
class GET_IMAGES:
#    Define function __init__(take 3 inputs from user= search_query, num_of_images, target directory with dir_)
    def __init__(self):
        self.search_query = input('Enter Search query(e.g: SEC):')
        self.number_of_images = int(input('Enter number of images:'))
        self.dir_ = input('Enter target directory(e.g: ..../..../SEC_Images/):')
#        if target directory does not exist it then make it
        if not os.path.exists(self.dir_): 
            os.mkdir(self.dir_)
#            calls Dounload_images function
        self.run = self.Download_images()
    
#   function to extract images and save it    
    def Download_images(self):
#        assigns given search query to query, target diectory to dir, and number_of_images to num_images
        query = self.search_query
        dir = self.dir_
        num_images = self.number_of_images
#        split given query string and then join to putin url
        query = query.split()
        query = '+'.join(query)
        url = "https://www.google.com/search?q="+query+"&sxsrf=ALeKk0195UkFFq8HncMlk5y78V4UeqWYoQ:1611767422846&source=lnms&tbm=isch&sa=X&ved=2ahUKEwja2svFzbzuAhXVQUEAHW7WBU8Q_AUoAXoECAIQAw&biw=1600&bih=705"
        # print(url)
#        Selenium browsing option object called from selenium library
        options = Options()
#        for headless browsing assign True value
        options.headless = True
#        opens brownser in headless mode
        driver = webdriver.Chrome(options=options)
#        browser takes url for searching
        driver.get(url)
        
#        page scrolls up after each 2 seconds
        page_scroll_sleep = 2
        
            # Get scroll height
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll down to bottom(from 0 height till to end) 
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
            # Wait to load page
            time.sleep(page_scroll_sleep)
        
            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
        
            if new_height == last_height:
            #break #insert press load more
                try:
                    element = driver.find_elements_by_class_name('mye4qd') #returns list
                    element[0].click()
                except:
                    break
#            breacks the while loop till new_height of page =last_height
            last_height = new_height
        
            ## gets link list of images by class tags
        image_links = driver.find_elements_by_class_name('n3VNCb')
#        define count
        count = 0
#        define empty list
        imagelinks= []
#        loop over image_links in res variable
        for res in image_links:
            try:
#                pick image attribute in each image_link and put in links
                link = res.get_attribute('src')
#                if any link have no attribute, remove
                if link!=None:
#                    put all not none links in the imagelinks list define above
                    imagelinks.append(link)
#                after each itteration count automatically increaments by 1
                count = count + 1
#                when count = number of requested images the condition stops
                if (count >= num_images):
                    break
                    
            except KeyError:
                continue
#        prints the founded links appended to be saved
        print(f'Found {len(imagelinks)} images')
        print('Start downloading...')
#        for each image link in imagelinks
        for imagelink in imagelinks:
            # open each image link and save the file with file name img(current time) in given target directory as JPEG format
            imagename = dir + '/' + 'img' + str((int(time.time()))) + '.JPEG'
#            retrieve image for the requested link of image
            urllib.request.urlretrieve(imagelink, imagename) 
#       prints Download completed when complete downloading
        print('Download Completed!')
        
#        then quit browser
        driver.quit()
#        return 1 if no error message
        return 

 # faster_rcnn/openimages_v4/inception_resnet_v2 model used from following url
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
# call detector from url through tensorflow hub and assign to detector
detector = hub.load(module_handle).signatures['default']

# Annotate Images
class IMAGE_ANNOTATER:
   
    def __init__(self):
#        takes path where images located
        self.path = input('Enter input images path:(eg: .../.../SEC_Images)')
#        calls the create_excel function
        self.create_excel()
        
#    function to draw boxes around objects in images
    def draw_boxes(self, image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
      lst = []
#      loop over minimum boxes shapes 
      for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >=.50:  # if threshold value is greater or equal than .50 then append True to lst
            lst.append(True) # for aesthetic
        else:
            lst.append(False) # if threshold value is less than .50 then appen False to lst for other
        check = any(lst) # check for true values and false values
        if check == True: # if True means aesthetic return 1
          result=1
        elif check==False: # if false means other return 0
          result= 0
        return result
#    used to label images
    def _label_image(self, image_path):
#        open images using path
      img = tf.io.read_file(image_path)
#      decode image in jpeg format with channels=3 means RGB colors
      img = tf.image.decode_jpeg(img, channels=3)
#     convert images array into float32 datatype
      converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
#      decode the converted image
      result = detector(converted_img)
#     assign each result value to key in numpy array and store in results
      result = {key:value.numpy() for key,value in result.items()}
#     pass parameters to draw_boxes function 
      image_with_boxes = self.draw_boxes(
          img.numpy(), result["detection_boxes"],
          result["detection_class_entities"], result["detection_scores"])
      return image_with_boxes
    def classify(self,image_path):
      # load the image
      image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
      # make copy of image
      output = image.copy()
      
      # resize image to 70x70
      image = cv2.resize(image, (70, 70))
      # get image pixel
      image = image.astype("float") / 255.0
      # convert image to array
      image = img_to_array(image)
      # expand image as size and channels
      image = np.expand_dims(image, axis=0)

      # load the trained convolutional neural network and the label
      # binarizer
      print("[INFO] loading network...")
      model = load_model('img_clf.model')
      lb = pickle.loads(open('label.pickle', "rb").read())
      # classify the input image
      # print("[INFO] classifying image...")
      proba = model.predict(image)[0]
      idx = np.argmax(proba)
      label = lb.classes_[idx]
      if proba[idx] >= .50:
        return label
      else:
        return 'None' 

    def calc_statistics(self, image_path):
#        image statistics using PIL and colorgram library
        image             = Image.open(image_path) # open image
        fullPath          = image.filename # return full path and filename
        img_cat = self.classify(image_path)
        doc = 0
        graph = 0
        signature = 0
        table = 0
        label = 0
        if img_cat == 'Document':
          doc = 1
          graph = 0
          signature = 0
          table = 0
          label = 0
        elif img_cat == 'Graph':
          doc = 0
          graph = 1
          signature = 0
          table = 0
          label = 0
        elif img_cat == 'Signature':
          doc = 0
          graph = 0
          signature = 1
          table = 0
          label = 0
        elif img_cat == 'Data_Table':
          doc = 0
          graph = 0
          signature = 0
          table = 1
          label = 0
        else:
          doc = 0
          graph = 0
          signature = 0
          table = 0
          label = self._label_image(image_path) # calls _label_image function and return 1s and 0s
        area              = image.size # return size of image
        size              = os.path.getsize(image_path) # get size of image in bytes
        format_           = image.format # return format of image
        mode              = image.mode  # return mode of image
        colors = colorgram.extract(image_path, 6) # return 6 color values of image
        color_list = [(color.rgb, color.proportion) for color in colors] # store 6 color values of image in list
        color_hsl = [(color.hsl, color.proportion) for color in colors]  # store 6 HSL color properties of image in list
        img               = ImageStat.Stat(image)
        eachbandpix       = img.count # return total pixes in each band of image
        eachbandpixsum    = img.sum   # return sum of each band pixels of image
        eachbandpixsqsum  = img.sum2  # return square sum of each band pixels of image
        eachbandpixavg    = img.mean  # return average of each band pixels of image
        eachbandpixmedian = img.median # return median of each band pixels of image
        eachbandrms       = img.rms    # return root mean square error of each band pixels of image
        eachbandvariance  = img.var    # return variance of each band pixels of image
        eachbandsdvar     = img.stddev # return standard diviation of each band pixels of image
        
#        store all stats of images in list
        lst = [fullPath,doc,graph,signature,table,label,area,size,format_,mode,color_list,color_hsl,eachbandpix,
           eachbandpixsum,eachbandpixsqsum,eachbandpixavg,
           eachbandpixmedian,eachbandrms,eachbandvariance,eachbandsdvar]
        return lst
    
    def create_excel(self):
#        column names of excel file
      keys = ["ImagePath",'Document','Graph','Signature','Table','Label',"Area(lxW)","Size_in_Bytes","Format","Mode","Color_List","Color_HSL","Each_Band_Pixels",
            "Each_Band_Pixel_Sum","Each_Band_Pixel_SquareSum","Each_Band_Pixel_Average",
            "Each_Band_Pixel_Median","Each_Band_RMS","Each_Band_Variance","Each_Band_SD_Variance"]
      with open('output.csv', 'w', newline = '') as output_csv:
        csv_writer = csv.writer(output_csv)
        csv_writer.writerow(keys)
        lst_imagePath = glob.glob(self.path + '/*.JPEG')
        for i in tqdm(range(len(lst_imagePath))):
            lst = self.calc_statistics(lst_imagePath[i])
            csv_writer.writerow(lst)
      return
