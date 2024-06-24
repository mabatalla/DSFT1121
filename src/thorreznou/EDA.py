#imports
    #resize_image()
from cv2 import imread , resize , IMREAD_GRAYSCALE
from os import remove , walk
from os.path import splitext
import zipfile
from pathlib import PurePath

    #reduce_image()
from cv2 import resize , INTER_AREA

    #process_color()
from numpy import floor

    #reduce_color_palette()
from numpy import reshape

import pandas as pd
import datetime as dt

# resize list of images to same sizes / dimensions
def resize_image(directory_path,
                pixels_1,
                pixels_2,
                bw = False,
                img_name_list= None,
                zip_file = False):
    '''   
    Resize images in .jpg format. Its purpose is to have all the image set levelled off 
    to use it in a convolutional neural net. Returns a list of the resized images.
    
    Keywords arguments:
    directory_path -- directory containing the set of original images
    pixels_1 -- value for the horizontal axis (pixels)
    pixels_2 -- value for the vertical axis (pixels)
    img_name_list -- in case there is a list of images names to use (default: None)
        
    '''
    
    #List to append the resized images
    re_image_list = []
    admited_formats_list = ['.bmp' , '.dib' , '.jpeg' , '.jpg' , '.jpe' , '.jp2',
                            '.png' , '.webp' , '.pbm' , '.pgm' , '.ppm' , '.pxm',
                            '.pnm' , '.sr' , '.ras' , '.tiff' , '.tif' , '.exr' ,
                            '.hdr' , '.pic']
    
    #If images in zip_file, extract and delete original zip
    if zip_file:
        
        with zipfile.ZipFile(directory_path , 'r') as z:
            
            for name in z.namelist():
                
                if (img_name_list != None):
                    z.extract(name, '.\images_unzip')
                else:
                    z.extract(name, '.\images_unzip')
                    
        #Delete zip file
        remove(directory_path)
        
        #Search paths in directories and subdirectories for all the images
        img_path_list = []
        for path , folders , files in walk('.\images_unzip'):
            
            #If not a predefined list of images, take all paths
            if img_name_list == None:
                
                for name in files: 
                    file_ext = splitext(name)[1]
                    
                    if file_ext.lower() in admited_formats_list:
                        img_path_list.append(PurePath(path, name))
                    else:
                        continue  
                    
            #If a predefined list of images, take only the ones in list
            if img_name_list != None:
                
                for name in files: 
                    
                    if name in img_name_list:
                        file_ext = splitext(name)[1]
                        
                        if file_ext.lower() in admited_formats_list:
                            img_path_list.append(PurePath(path, name))
                        else:
                            continue

        #Resize images and append to list
        print('Nº images:' ,
              len(img_path_list), '------- Final shape: %sx%s pixels' % (pixels_1 , pixels_2))
        
        for image_path in img_path_list:
            if bw:
                img = imread(str(image_path) , flags=IMREAD_GRAYSCALE)
            else:
                img = imread(str(image_path))
            image = resize(img, (pixels_1, pixels_2))
            re_image_list.append(image)     
        
    #If not zip_file
    if not zip_file:
        
        img_path_list = []
        for path , folders , files in walk('.\images_unzip'):
            
            #If not a predefined list of images, take all paths
            if img_name_list == None:
                
                for name in files: 
                    file_ext = splitext(name)[1]
                    
                    if file_ext.lower() in admited_formats_list:
                        img_path_list.append(PurePath(path, name))
                    else:
                        continue  
                    
            #If a predefined list of images, take only the ones in list
            if img_name_list != None:
                
                for name in files: 
                    
                    if name in img_name_list:
                        file_ext = splitext(name)[1]
                        
                        if file_ext.lower() in admited_formats_list:
                            img_path_list.append(PurePath(path, name))
                        else:
                            continue
                        
        #Resize images and append to list            
        print('Nº images:' , 
              len(img_name_list), '------- Final shape: %sx%s pixels' % (pixels_1 , pixels_2))
        
        for image_path in img_path_list:
            if bw:
                img = imread(str(image_path) , flags=IMREAD_GRAYSCALE)
            else:
                img = imread(str(image_path))
            image = resize(img, (pixels_1, pixels_2))
            re_image_list.append(image)
        
    return re_image_list


# standarize list of numbers into a format which makes possible to convert to float / integers
def standarize_numbers(numbers_list, 
                        type):
    '''
    Convert a list of numbers in string format to float format in case of 
    incompatibilities. Returns a list of float numbers (for "decimal" and "double") 
    or a list of integers (for "integer").
    
    Keywords arguments:
    numbers_list -- list of elements to convert
    type -- number type to convert. 3 values:
        - "decimal": decimal number denoted by "," instead of ".". Ej.: 6,67
        - "integer": integer number with thousand, millions, billions, etc. denoted by '.'. Ej.: 12.000.000
        - "double": number written in both previous forms. Ej.: 6.000.000,255 
    '''
    
    #Make transformations by 'type' argument value
    if type == 'decimal':
        float_list = list(map(lambda num: float(num.replace(',' , '.')) , numbers_list))
    elif type == 'integer':
        float_list = list(map(lambda num: int(num.replace('.' , '')) , numbers_list))
    elif type == 'double':
        float_list = list(map(lambda num: float(num.replace('.' , '').replace(',' , '.')) ,
                              numbers_list))
    else:
        print('Type not valid')

    return float_list


# transform str decimal coordenates into float decimal coordenates
def coordinates(list_lat,
                list_lng,
                tuple_format= False):
    '''
    Function to transform mixed coordinates (integers + cardinal point) to decimal 
    coordenates. Be careful, the two firts numbers will be taken as the integer part 
    of the decimal coordinate. Return 2 options: separated lists for latitude and 
    longitude or list of tuples representing geographic points.
    
    Keyword arguments:
    list_lat -- list of latitude values. Ej.: 400782N , 031213s 
    list_lng -- list of longitud values. Ej.: 239872e , 328769W (also accepts Spanish version "O")
    tuple_format -- If False, returns separated lists; if True, returns a list of tuples (default False)
    '''
    
    #List to append new latitude values and count to show position in case of error
    lat = []
    contador = 0
    
    #Loop to take all values in original latitude list
    for coor in list_lat:
        #Check if the original value is correctly written, if not, adds '0' before value to avoid errors
        if len(coor) < 7:
            coor = (7-len(coor))*'0' + coor
            
        #North values
        if (coor[-1] == 'N') or (coor[-1] == 'n'):
            valor = coor[0 : 2] + '.' + coor[2 : -1]
            lat.append(float(valor))
            #South values
        elif (coor[-1] =='S') or (coor[-1] =='s'):
            valor = coor[0 : 2] + '.' + coor[2 : -1]
            lat.append(-float(valor))
        #Anything else
        else:
            print('Latitudinal coordenate (%s) not valid in position (%s)' % (coor , contador))
            break
        
        contador += 1
        
    #List to append new longitude values and count to show position in case of error
    lng = []
    contador = 0
    #Loop to take all values in original longitude list
    for coor in list_lng:
        #Check if the original value is correctly written, if not, adds '0' before value to avoid errors
        if len(coor) < 7:
            coor = (7-len(coor))*'0' + coor
            
        #West values
        if (coor[-1] == 'W') or (coor[-1] == 'w') or (coor[-1] == 'O') or (coor[-1] == 'o'):
            valor = coor[0 : 2] + '.' + coor[2 : -1]
            lng.append(-float(valor))
        #East values
        elif (coor[-1] == 'E') or (coor[-1] == 'e'):
            valor = coor[0 : 2] + '.' + coor[2 : -1]
            lng.append(float(valor))
        else:
            print('Longitudinal coordenate (%s) not valid in position (%s)' % (coor , contador))
            break
        
        contador += 1
    
    #If taple_format, return list of coordenate tuples with zip(). If not, return separated lists
    if tuple_format:
        coor = list(zip(lat , lng))
        
        return coor
    
    else:
        
        return lat , lng


# reduce image size maintaining the original proportions
def reduce_img(image, height):
    '''
    Reduce the image dimensions, maintaining the ratio between original width 
    and height values. Returns the reduced image.
    
    Keyword arguments:
    image -- image to reduce dimensions
    height -- new height
    '''
    
    #Set dimensions ratio
    ratio = image.shape[0]/image.shape[1]
    #Calculate new width
    width = int(height/ratio)
     
    return resize(image, (width, height), interpolation=INTER_AREA)      


# process color image to use in the 'reduce_color_palette' function
def process_color(channel_value, bins):
    '''
    This function is used to map the values of RGB in a pixel that
    go from 0 to 255 to a simple version with X values that lead to
    a palette of Y colors (Xx Xx Xx = Y).
    Eg. X = 4, then y = 4x4x4 = 64 colors
    
    It is used in 'reduce_col_palette' function
    
    Keywords arguments:
    channel_value -- value of color intensity for each pixel in the BGR layers
    bins -- number of colours to reduce each layer (total colors = bins^3)
    '''
    
    if channel_value >= 255: 
        processed_value = 255
    else:
        preprocessed_value = floor((channel_value*bins)/255)
        processed_value = abs(int(preprocessed_value*(255/(bins-1))))
    
    return processed_value


# reduce the number of colours of a image
def reduce_col_palette(image, bins, info=False):
    '''
    This function iterate through every pixel of an image to map
    each rgb channel color value to a reduced palette.
    
    Keywords arguments:
    image -- image to reduce color palette
    bins -- number of colours to reduce each layer (total colors = bins^3). Used in 'process_color' function
    info -- if True, prints a message with the number of color of the new image
    '''
    
    # Capture image dimensions
    img = image.flatten()
       
    # Iterate the array to transform the value of the pixels
    for px in range(len(img)):
        
        if img[px] == 255: img[px] = 255
        else: img[px] = process_color(img[px], bins)
    
    # Restore image shape
    img = reshape(img, image.shape)
        
    # Inform user if 'info' is True about the total number of colors in new image
    if info:
        print(f'Palette reduced to {bins**3} colors.')
            
    return img


# calculate math expectation of a set of features
def math_expect(data, column_name_groups= None, column_name_success= None, n_top= 100):
    '''
    Function that calculates the mathematical expectation of a random variable ('column_name_groups').
    Definition: Mathematical expectation, also known as the expected value, is the the sum or integral of all possible
    values of a random variable, or any given function of it, multiplied by the respective probabilities of the values 
    of the variable. Return the DataFrame used as input containing a new column named 'math_expectation' 
    with the resulting values.
    
    Keyword arguments
    
    data (Pandas DataFrame type) -- is the given dataframe
    column_name_groups (string) -- name of column with the groups (random variable) to calculate the math expect
    column_name_success (string) -- name of column that sets the success measurement. ie: daily income or scoring 
    from movies reviews    
    n_top (integer) -- default 100, number of occurrencies within the top shortlist. ie: top 100, top 1000 or top 10
    '''
    len_data = len(data)

    #WE GROUP BY THE COLUMN WITH THE CHOSEN RANDOM VARIABLE
    groups = pd.DataFrame(data.groupby(column_name_groups)[column_name_groups].count())
    groups.rename(columns={column_name_groups:'groups'}, inplace=True)
    groups.reset_index(inplace=True)

    #WE ADD A COLUMN WITH THE DISTRIBUTION OF EACH GROUP IN %
    groups['% per group'] = (groups['groups'] / len_data) * 100

    #WE CREATE A COLUMN TO SEE THE EXPECTED NUMBER PER GROUP IN THE TOP LIST
    groups['expected_in_top'] = (groups['% per group'] * 10)

    #WE SORT BY THE SUCCESS VARIABLE UNTIL THE N_TOP FIGURE. IE: TOP 10, TOP 100, TOP 1000, ETC
    data.sort_values(by=column_name_success, ascending=False, inplace=True)
    top = data[:n_top]

    #WE GROUP BY AND CALCULATE THE ACTUAL NUMBER OF EACH GROUP VARIABLE IN THE TOP LIST
    top = top.groupby(column_name_groups)[column_name_groups].count()
    top.sort_values(ascending=False, inplace=True)
    top = pd.DataFrame(data=top)
    top.rename(columns={column_name_groups:'actual_top'}, inplace=True)
    top.reset_index(inplace=True)

    #WE MERGE THE TWO DATAFRAMES
    #WE REPLACE THE NAN VALUES FOR ZEROS
    join_group_top = groups.merge(top, how='left', on=column_name_groups)
    join_group_top['actual_top'] = join_group_top['actual_top'].fillna(0)

    #WE CREATE THE MATH_EXPECTATION COLUMN WITH ITS FORMULA AND SORT IT OUT BY THAT COLUMN
    join_group_top['math_expectation'] = ((join_group_top['actual_top'] - join_group_top['expected_in_top']) / join_group_top['expected_in_top'])*100
    join_group_top.sort_values(by='math_expectation', ascending=False, inplace=True)

    #WE CREATE A DICTIONARY WITH THE GROUP NAMES AND MATH_EXPECTATION VALUES
    dicc = {}
    for i, j in zip(join_group_top[column_name_groups], join_group_top['math_expectation']):
        dicc[i] = j

    #WE MAP THE RESULTS AND SORT BY MATH_EXPECTATION COLUMN
    data['math_expectation'] = data[column_name_groups].map(dicc) 
    data.sort_values(by='math_expectation', ascending=False, inplace=True)

    return data


# dataframe overview info
def overview(df):
    '''
    Function that summarizes data in a new dataframe from an originally given dataframe. It returns a new DataFrame
    with valuable information from the original DataFrame you are working on.
  
    Keyword arguments
    data (Pandas DataFrame type) -- the given dataframe
    '''
    #COLUMN NAMES
    cols = pd.DataFrame(df.columns.values,columns=['column names'])
    
    #COLUMN TYPES
    types = pd.DataFrame(df.dtypes.values, columns=["type of data"])

    #NAN TOTAL
    total_nan = df.isna().sum()
    total_nan = pd.DataFrame(total_nan.values, columns = ["nan total"])

    #NAN %
    percent_nan = round(df.isna().sum()*100/len(df),2)
    percent_nan_df = pd.DataFrame(percent_nan.values, columns = ["nan %"])

    #MISSINGS %
    percent_missing = round(df.isnull().sum()*100/len(df),2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns = ["missing %"])
 
    #UNIQUE VALUES
    unique = pd.DataFrame(df.nunique().values, columns = ["unique values"])
    
    #CARDINALITY
    percent_cardin = round(unique["unique values"]*100/len(df),2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns = ["cardin %"])

    #WE CONCAT THE DATAFRAMES
    concatenate = pd.concat([cols, types, total_nan, percent_nan_df, percent_missing_df, unique, percent_cardin_df], axis=1)
    concatenate.set_index('column names', drop=True, inplace=True)

    #WE ADD THE DESCRIBE() FUNCTION RESULTS AND CONCATENATE THEM TO THE PREVIOUS DATAFRAME
    df_describe = pd.DataFrame(df.describe(include= 'all'), columns=df.columns)
    df_concat = pd.concat([concatenate.T, df_describe])

    return df_concat


# removes outliers from a feature and related entries
def outlier_removal(data, column_name= None, p_max=None, p_min=None):
    '''
    Function that removes outliers from a given dataframe, specifying the column.
    min and max percentiles are passed as parameters to define the line between outliers and non-outliers.
    Returns a new DataFrame without the data below the p_min and above the p_max.
    
    Keyword arguments
    data (Pandas DataFrame type) -- is the given dataframe
    column_name (string) -- the chosen column to detect and remove outliers from
    p_max (integer) -- above this percentile we start removing outliers
    p_min (integer) -- below this percentile we start removing outliers
    '''
    q_max = p_max / 100
    q_min = p_min / 100
    q_max_column = data[column_name].quantile(q_max)
    q_min_column = data[column_name].quantile(q_min)

    data = data[(data[column_name] > q_min_column) & (data[column_name] <= q_max_column)]
    return data 


# transform columns to separated data values
def datetime_into_columns(df, column, weekday = False, hour_minutes = False, from_type = 'object'):
    """ 
    The function converts a column with a date from either int64 or object type
    into separate columns with day - month - year
    user can choose to add weekday - hour - minutes columns

    Keyword arguments
    df (Pandas DataFrame type)-- is the given dataframe
    column (string) -- the chosen column to create new columns from
    weekday (boolean) -- True if user wants new column with weekday value (default False)
    hour_minutes (boolean) -- True if user wants two new columns with hour and minutes values (default False)
    from_type (string) -- 'object' by default if original column type is object and
                        'int64' if original column type is int64

    return: the resulting dataframe with the new colum
    """

    if from_type == 'int64':
        column  = pd.to_datetime(df[column].astype(str))
    else:
        column  = pd.to_datetime(df[column])
        
    datetime = pd.DataFrame(column)

    datetime['day'] = column.dt.day
    datetime['month'] = column.dt.month
    datetime['year'] = column.dt.year

    if weekday == True:
        datetime['weekday'] = column.dt.weekday

    if hour_minutes == True:
        datetime['hour'] = column.dt.hour
        datetime['minutes'] = column.dt.minute

    df = pd.concat([df, datetime], axis = 1)
    df = df.loc[:,~df.columns.duplicated(keep='last')]
    
    return df


# treatment of missing data (NaNs) by column
def missing_data(df, columns = None, method = 'delete'):
    """
    The function allows user to decide whether they want to delete, convert to zeros, or replace by the 
    mean / mode / median all missing values in a specific list of columns of the given df. Returns the 
    resulting dataframe.

    Keyword arguments
    df (Pandas DataFrame type) -- is the given dataframe
    columns (list) -- the chosen column/s with missing values
    method (string) -- how to treat the NaN values
        - 'delete' drops the missing values (by default)
        - 'zero' fills missing values with a zero
        - 'mean' fills missing values with the mean
        - 'mode' fills missing values with the mode
        - 'median' fills missing values with the median
    """
 
    if method == 'delete':
        df.dropna(subset=columns, inplace= True)    
    elif method == 'zero':
        
        for column in columns:
            df[column] = df[column].fillna(0)
            
    elif method == 'mean':
        
        for column in columns:
            df[column] = df[column].fillna(df[column].mean())
            
    elif method == 'mode':
        
        for column in columns:
            df[column] = df[column].fillna(df[column].mode()[0])
            
    elif method == 'median':
        
        for column in columns:
            df[column] = df[column].fillna(df[column].median())
            
    return df
