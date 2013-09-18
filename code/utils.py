import random
import numpy as np
import datetime
import random
from numpy import where, nonzero

from dateutil import parser, relativedelta
from datetime import datetime


def MakeConverters(headers):
    '''Numpy's loadtxt uses 'converters' to cast data to floats.
    This function generates a converter for each column.'''
    converters = dict()
    for i in range(len(headers)):
        converters[i] = lambda val: Converter(val)
    return converters

   
def Converter(value):
    '''If value cannot be immediately converted to a float, then return a NaN.'''
    try: return float(value or 'nan')
    except ValueError: return np.nan

 
def Draw(count, max):
    '''draw random numbers without replacement'''
    result = []
    iterations = range(count)
    for i in iterations:
        index = int(max * random.random())
        if index in result: iterations.append(1)
        else: result.append(index)

    return result
 

def Quantile(list, q):
    '''Find the value at the specified quantile of the list.'''
    if q>1 or q<0:
        return np.nan
    else:
        list = np.sort(list)
        position = np.floor(q * (len(list)-1) )
        
        #if len(list) > position+1 : position += 1
        
        return list[position]
        

def NonStandardDeviation(list, pivot):
    var = 0
    for item in list:
        var = var + (item-pivot)**2
        
    return np.sqrt( var/len(list) )

    
def ImportFile(file_name):
    '''Import a ULP-type data file'''

    #open the data source file and read the headers
    infile = file_name
    f = open(infile, 'r')
    headers = f.readline().rstrip('\n').split(',')
    
    #strip the blank headers and the 'Date' header
    headers = filter(lambda x: x!='', h)
    headers = flatten(['year', 'julian', headers[1:]])
    
    #define a couple of objects we'll use later on.
    data = list()
    finished = False
    
    #loop until the end of the file:
    while not finished:
        line = f.readline()
        
        #continue unless we're at the end of the file
        if not line:
            finished = True
        else:
            values = line.rstrip('\n').split(',')
            
            #only process data that has some value in the first field
            if not values[0]: pass
            else:
                #get only those columns with a valid header
                v = np.array(values)
                v = v[indx]
                values = list(v)
                
                #convert the date into our expected form
                date_obj = ObjectifyDate(values[0])
                julian = Julian(date_obj)
                
                #flatten the data into a numpy array (from a list of lists)
                data_row = flatten([date_obj.year, julian, values[1:]])
                data_row = np.array(data_row)
                
                #add this row of data to the big list.
                data.append(data_row)

    #Remove blank rows:
    data = filter(lambda x: '' not in x, data)
    
    #make the big list into a big array
    data = np.array(data)
    data = data.astype(float)
    
    return [headers, data]


def Factorize(data, target='year', headers='', levels='', return_levels=False):
    '''Turn target into a factor whose coefficients will sum-to-zero'''

    #if the data is not originally a dictionary, turn it into one.
    if type(data) is dict:
        passed_dict = True
    else:
        data = dict( zip(headers, data.transpose()) )
        passed_dict = False

    original = data.pop(target)

    #Decide what levels the factor can take (prepend the column name to make sure levels are strings)
    if not levels:
        levels = [ target+str(int(i)) for i in np.unique(original)]

    #If levels are passed  to the function, then code the factor into those levels
    else: pass

    #Produce the ways to code this factor (number of columns is one fewer than the number of levels. The last level is coded -1,...,-1)
    labels = np.diag( np.ones(len(levels)-1) )
    labels = np.vstack( (labels, -1*np.ones(len(levels)-1)) )
    missing = np.zeros(len(levels)-1)

    factorized = list()

    #Code each observation appropriately.
    for obs in original:
        if target+str(obs) in levels: factorized.append( labels[levels.index(target+str(obs)),:] )
        else: factorized.append( missing )

    factorized = np.array(factorized)    

    #Add the new columns to the data dictionary.
    for level in levels[:-1]:
        data[ str(level) ] = factorized[:,levels.index(level)].squeeze()

    #If the data was passed as an array, return an array
    if not passed_dict:
        data_array = np.array(data.values()).transpose()
        headers = data.keys()

        if not return_levels: return [headers, data_array]
        else: return [headers, data_array, levels]

    #Otherwise, return a dictionary.
    else:
        if not return_levels: return data
        else: return [data, levels]


def ReadCSV(file, NA_flag=-99999):
    '''Read a csv data file and return a list that \nconsists of the column headers and the data'''
    infile = file

    headers = open(infile, 'r').readline().lower().rstrip('\n').split(',')
    data_in = np.loadtxt(fname=infile, skiprows=1, dtype=float, unpack=False, delimiter=',', converters=MakeConverters(headers))
        
    #remove any rows with NA's
    nan_rows = np.nonzero(np.isnan(data_in))[0]
    mask = np.ones( data_in.shape[0], dtype=bool )
    mask[nan_rows]=False
    data_in = data_in[mask,:]
    
    #Now remove rows with NA_flags
    data_in = filter(lambda x: NA_flag not in x, data_in)
    data_in = np.array(data_in)

    return [headers, data_in]

    
def WriteCSV(array, columns, location):
    '''Creates a .csv file out of the contents of the array.'''
    out_file = open(location, 'w')
    
    for item in range( len(columns) ):
        out_file.write(columns[item])
        if item < len(columns)-1: out_file.write(',')
        else: out_file.write('\n')
        
    for row in range( array.shape[0] ):
        for item in range( array.shape[1] ):
            out_file.write( str(array[row,item]) )
            if item < array.shape[1]-1: out_file.write(',')
            else: out_file.write('\n')
    
    out_file.close()


def Partition(data, folds):
    '''Partition the data set into random, equal-sized folds for cross-validation'''
    
    #If we've called for leave-one-out CV (folds will be like 'n' or 'LOO' or 'leave-one-out')
    if str(folds).lower()[0]=='l' or str(folds).lower()[0]=='n' or folds>data.shape[0]:
        fold = range(data.shape[0])
    
    #Otherwise, randomly permute the data, then use contiguously-permuted chunks for CV
    else:
        #Initialization
        indices = range(data.shape[0])
        fold = np.ones(data.shape[0]) * folds
        quantiles = np.arange(folds, dtype=float) / folds
        
        #Proceed through the quantiles in reverse order, labelling the ith fold at each step. Ignore the zeroth quantile.
        for i in range(folds)[::-1][:-1]:
            fold[:Quantile(indices, quantiles[i])+1] = i
            
        #Now permute the fold assignments
        fold = np.random.permutation(fold)
        
    return fold


def Split(data, headers, year):
    '''Partition the supplied data set into training and validation sets''' 
    
    #model_data is the set of observations that we'll use to train the model.
    model_data = data[ where(data[:,headers.index('year')]<year)[0], : ]
    
    #validation_data is the set of observations we'll use to test the model's predictive ability.
    validation_data = data[ where(data[:,headers.index('year')]==year)[0], : ]
    
    model_dict = dict(zip(headers, np.transpose(model_data)))
    validation_dict = dict(zip(headers, np.transpose(validation_data)))

    return [model_data, validation_data]


def ObjectifyTime(time_string):
    '''Create a time object from from a time string'''
    [hour, minute, second] = time_string.split(':')
    time_obj = datetime.time(hour=hour, minute=minute, second=second)
    return time_obj
 
 
def ObjectifyDate(date_string):
    '''Create a date object from from a date string'''
    try:
        #Try the easy way first
        date_obj = parser.parse(date_string)
        
    #Tokenize the string and make sure the tokens are integers
    except ValueError:
        try:
            date_string.index('/')
            values = date_string.split('/')
        except ValueError:
            date_string.index('.')
            values = date_string.split('.')
            values = map(int, values)
        
        #Create and return the date object
        try:
            date_obj = datetime.date(month=values[0], day=values[1], year=values[2])
        except ValueError:
            date_obj = datetime.date(month=values[1], day=values[2], year=values[0])
            
    return date_obj
 
 
def Julian(date):
    '''Get the number of days since the start of this year'''
    year_start = datetime.date(month=1, day=1, year=date.year)
    julian = (date.date() - year_start).days + 1
    return julian

 
def MatchDictionaries(dict1, dict2):
    '''Create a new dictionary that combines data from the matching keys of two separate dictionaries'''
    dict_matched = dict()
    
    for key in dict1.keys():
        if key in dict2:
            val_list = [list(dict1[key]), list(dict2[key])]
            val_list = flatten(val_list)
            dict_matched[key] = val_list

    return dict_matched
    

def MatchData(struct1, struct2):
    '''Create a new data structure from the matching headers of two separate data structures'''
    
    #unpack the parameters
    [headers1, array1] = struct1
    [headers2, array2] = struct2
    
    #create new lists that we will fill
    matched_headers = list()
    matched_values = list()
    
    #find the headers that appear in both structures
    for col in headers1:
        if col in headers2:
            #combine data from the matching columns
            val_list = [list(array1[:,headers1.index(col)]), list(array2[:,headers2.index(col)])]
            val_list = flatten(val_list)
            
            matched_headers.append(col)
            matched_values.append(val_list)
    
    #turn the combined data into an array and return it with the headers
    matched_values = np.transpose( np.array(matched_values) )
    return [matched_headers, matched_values]
    
    
def flatten(x):
    '''flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]'''

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

    
def ProbabilityOfExceedance(prediction, threshold, se):
    r = rdn.Wrap()
    return r.Call(function='pnorm', q=(prediction-threshold)/se).AsVector()[0]