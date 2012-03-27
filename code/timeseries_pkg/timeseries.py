import numpy as np
import datetime
from matplotlib import dates, mlab
from dateutil import parser
import timestamp
import copy
import netcdftime 
import scikits.statsmodels as sm
import pywt
import utils
        

def Get_Input(data_source, **args):
    #Read a time series from the source and timestamp each observation.
    
    if 'NA_flag' in args:
        NA_flag = args['NA_flag']
    else: NA_flag = -1.23e25
    
    if 'dayfirst' in args:
        dayfirst = args['dayfirst']
    else: dayfirst = False
    
    if 'yearfirst' in args:
        yearfirst = args['yearfirst']
    else: yearfirst = False

    if 'columns' in args:
        columns = args['columns']
    else: columns = None
    
    #Establish the data source and read the headers
    source = Source(data_source)
    headers = source.headers
    
    #Now use the headers to figure out how we're going to turn the date/time columns into a timestamp.
    stamper = timestamp.Stamper(headers, dayfirst=dayfirst, yearfirst=yearfirst)
    
    #We will remove the columns that are used to generate the timestamp.
    cols = range( len(headers) )
    for c in stamper.time_columns:
        cols.remove(c)
    headers = list( np.array(headers)[cols] )
    
    #define a few of objects we'll use later on.
    timestamps = list()
    data = data = list()
    finished = False
    i=0
    
    #loop until the end of the file:
    while not finished:
        line = source.Next()

        #continue unless we're at the end of the file
        if type(line) == float: #should pick up a nan (type 'list' is good)
            finished = True
        else:
            #generate a timestamp
            stamp = stamper.Parse(line)
            timestamps.append(stamp)
            
            #filter out the columns we used to generate the timestamp
            data_row = []
            for c in cols:
                data_row.append( float(line[c] or np.nan) )
            
            #add this row of data to the big list.
            data_row = np.array(data_row)
            data.append(data_row)
   
   
    #make the big list into a big array
    data = np.array(data, dtype=float)
    data_shape = data.shape
    
    #eiminate any values flagged as NaN
    data = data.flatten()
    NAs = mlab.find(data == NA_flag)
    data[NAs] = np.nan
    data.shape = data_shape

    #if we've specified columns to keep, then keep those
    if columns != None:
        cols = mlab.find([c.lower() in h.lower() for c in np.array(columns, ndmin=1) for h in headers]) % len(headers)
        if cols.shape == (0,): cols=columns
    else: cols = np.arange( len(headers) )

    data = data[:,cols]
        

    #remove any rows with NaN's
    data_length = len(data)
    data_width = np.shape(data)[1]
    nan_rows = np.unique( mlab.find( np.isnan(data) ) // data_width )
    
    rows = np.array( range(data_length) )
    mask = np.ones( len(rows), dtype=bool )
    mask[nan_rows] = False
    
    data = data[mask,:]
    timestamps = np.array(timestamps)[mask]
    
    #set up the object data
    series = {}
    series['timestamps'] = np.array(timestamps)
    series['data'] = data
    series['headers'] = list( np.array(headers)[cols] )      
    
    return series       
    
    
def Seconds(interval):
    #Compute the number of seconds between begin and end.
    
    #time_dict gives the length, in seconds, of various intervals:
    time_dict = {'days':86400, 'seconds':1}
    
    seconds = 0
    
    for property in time_dict:
        seconds += getattr(interval, property) * time_dict[property]
    
    return seconds
    
    
    
def Sweep(series, period, begin=0, end=0):
    #Break the time series into <interval>-sized chunks.
    
    #Establish the variables we'll use
    begin = series.timestamps[0]
    end_interval = begin + period
    end = series.timestamps[-1]
    result = []
    
    #Extract each interval
    while end_interval < end:
        result.append(series.Extract_Range(begin, end_interval))
        begin = end_interval
        end_interval += period

    #Extract the last (incomplete) interval.
    result.append(series.Extract_Range(begin, end_interval))
    
    return Wrapper(series=result)
    
        

class Wrapper:
    '''Wraps several time series objects'''
    
    def __init__(self, **args):
        #Create a new time series wrapper
            
        #If we've passed a file name, open that file and read it in as a time series.
        if 'file' in args:
            raw = Get_Input(args['file'], **args)
            self.Wrap_Raw(raw)
            
        if 'series' in args:
            self.series = args['series']
            self.headers = self.series[0].headers
            self.Split()
            self.series = self.Apply('Fill_Gaps')
            
        #If we've passed a data dictionary, extract the timeseries from that.
        if 'dict' in args:
            raw = Get_Input(args['dict'], **args)
            self.Wrap_Raw(raw)
        
        
            
    def Wrap_Raw(self, raw_data, **args):
        #Create a time series object from the raw data.
        self.series = [Series(series=raw_data)]
        self.headers = self.series[0].headers
        
        if 'indifference' in args:
            i = args['indifference']
        else: i = 5    
            
        self.Split(indifference=i)
        self.series = self.Apply('Fill_Gaps')
    
    
    
    def Split(self, indifference=10):
        #Split the series into subseries, imputing values to fill gaps up to <indifference>
        
        i=0
        
        while i < len(self.series):
            series = self.series[i]
            begin = series.timestamps[0]
            j=0
            
            while j < len(series.breaks):
                b = series.breaks[j]
                missing = Seconds(series.timestamps[b] - series.timestamps[b-1])/Seconds(series.period) -1
                if missing > indifference:
                    self.series.insert( i+j+1, series.Extract_Range(begin, series.timestamps[b-1]) )
                    begin = series.timestamps[b]
                j += 1
    
            self.series.insert(i+j+1, series.Extract_Range(begin, series.timestamps[-1]) )
            self.series.pop(i)
            i += j + 1
            
    
    def Apply(self, method, **args):
        #Run method(**args) or each time series in the wrapper.
        
        return_list = []
        
        for s in self.series:
            m = getattr(s, method)
            try: return_list.append( m(**args) )
            except: pass

        return return_list
        
        
    def Stitch(self):
        #Combine the wrapped series into one.
        
        data = []
        timetamps = []
        
        
        for s in self.series:
            timestamps.extend(s.timestamps)
            data.extend(s.data)
            
        stitched = {}
        stitched['headers'] = self.headers
        stitched['timestamps'] = timestamps
        stitched['data'] = data
        
        self.series = [Series(series=stitched)]
        
        
    def ARX(self, model_args):
        #Generate an ARX model to fit all the data.

        #Generate the X matrix for the AR model
        matrices = self.Apply('ARX_Model_Matrix', args=model_args)
        model_args['matrix_generator'] = 'ARX_Model_Matrix'
         
        #Drop the imputed rows before fitting the model
        X = np.vstack( matrices )
        Y = np.concatenate( self.Apply('Get_Column', column=model_args['target'], strip_imputed=True) )
            
        #return the Model object (fitted by least squares)
        return Model(X,Y, model_args)
        
        
    def Spin_Off(self, column):
        #Returns a new time series wrapper that contains the <column> values from all subseries.
        
        spin_off_series = self.Apply('Spin_Off', column=column)
        return Wrapper(series=spin_off_series)
        
            



    
class Series:
    '''An object of this class represents a univariate time-series data stream'''
    
    def __init__(self, **args):
        #Create a new time series
             
        #If we've passed a file name, open that file and read it in as a time series.
        if 'file' in args:
            series = Get_Input(args['file'], **args)
            #series = Read_File(args['file'], **args)
            
        #If we've provided a series, then use that to create a new Series object.
        elif 'series' in args:
            series = args['series']
            
        self.headers = series['headers']
        self.data = series['data']
        self.timestamps = series['timestamps']
        
        #Mark which observatons were measured and which were imputed
        if 'imputed' in series: self.imputed = series['imputed']
        else: self.imputed = np.zeros( len(self.timestamps), dtype=bool )
        
        #Make sure the data array is at 2-dimensional and that headers is a list
        if self.data.ndim == 1:
            self.data.shape = self.data.shape[0], 1
            self.headers = [self.headers]
        
        #Put the observations in time order.
        self.Order_By_Time()
        
        #Find the measurement period and identify any breaks in the data stream:
        self.Remove_NaNs()
        self.Get_Period()
        self.Get_Breaks()
                  
    
    def Order_By_Time(self):
        #Rearrange the observations in time order.
        
        #Find the time order of the observations
        order = np.argsort(self.timestamps)
        
        #Now order the observations and their timestamps from first to last.
        self.timestamps = self.timestamps[order]
        
        self.data = self.data[order]
    
    
    def Get_Period(self):
        #Determine the sampling period for time series observations
        
        #Get the timestamps of the first twenty observations
        first_twenty = self.timestamps[0:20]
                
        #Take the smallest measurement period among the first twenty observations as the
        #fundamental period of the time series.
        period_list = []
        i = 1
        
        while i < len(first_twenty):
            difference = first_twenty[i] - first_twenty[i-1]
            period_list.append(difference)
            i += 1 #next row
            
        if len(period_list) > 1: self.period = min(period_list)
        else: self.period = 0
        
        
    def Get_Breaks(self):
        #Count the spots where time series measurements are interrupted, or when all observations are NaN
        
        #Begin with a blank list of breaks
        self.breaks = breaks = []
        
        #Check whether each row indicates a gap
        i = 1
        while i < len(self.timestamps):
            diff = self.timestamps[i] - self.timestamps[i-1]
            if diff != self.period: breaks.append(i)
            i += 1
    
    
    def Remove_NaNs(self):
        #Remove from the time series those rows where all observations are NaNs.
        
        #Begin with a blank list of non-blank rows
        not_blank = []
        
        #Check whether each row is all NaNs
        for i in range( len(self.timestamps) ):
        
            if sum(np.isnan(self.data[i])) != self.data.shape[1]:
                not_blank.append(i)
                
        self.data = self.data[not_blank,:]
        self.timestamps = self.timestamps[not_blank]
                
       
    def Get_Longest(self):
        #Trim the time series to be the longest uninterrupted sequence
        
        #Find the length of each continuous sequence
        intervals = []
        last = 0
        for d in self.breaks:
            intervals.append(d-last)
            last = d
            
        #From the last break to the end is also an interval
        intervals.append(self.data.shape[0] - last)
        
        #Find the longest uninterrupted series among the list of discontinuities
        location = np.argmax(intervals)
                
        #Figure out the beginning and end of the longest streak, which might appear at the beginning or end of the time series.
        if location==0: begin = 0
        else: begin = self.breaks[location-1]
        
        if location == len(intervals)-1: end = self.data.shape[0]
        else: end = self.breaks[location]
        
        #Create a new Series object from the range.
        return self.Extract_Range(self.timestamps[begin], self.timestamps[end-1])
        
        
    def Get_Column(self, column, **args):
        #Return the values from a particular column.
        index = self.Get_Index(column)
    
        #if we've been asked to strip the imputed values, then do so
        if 'strip_imputed' in args:
            if args['strip_imputed']==True:
                good_rows = mlab.find(self.imputed == False)
                return self.data[good_rows, index]
         
        #Otherwise, return all data 
        else: return self.data[:,index]
        
        
    def Extract_Range(self, begin, end):
        #Extract a date range and use it to make a new time series.
        
        try: 
            datetime.datetime.now() > begin
        except TypeError:
            begin = parser.parse(begin)
            
        try: 
            datetime.datetime.now() > end
        except TypeError:
            end = parser.parse(end)
        
        #Find which rows of timestamps lie in this range.
        upper_rows = mlab.find( self.timestamps >= begin )
        lower_rows = mlab.find( self.timestamps <= end )
        rows = filter( lambda x: x in upper_rows, lower_rows )

        #Now create a time series with the data in this range.
        range = {}
        range['headers'] = self.headers
        range['timestamps'] = self.timestamps[rows]
        range['data'] = self.data[rows,:]
        
        return Series(series=range)
        
        
        
    def Fill_Gaps(self, begin=None, end=None):
        #Fill any gaps in the data stream with the last value measured before the gap.
        
        #Get working copies
        timestamps = list(self.timestamps)
        data = copy.copy(self.data)
        imputed = list(self.imputed)
        breaks = copy.copy(self.breaks)
        breaks.reverse()
        
        #For each gap in the data stream...
        for b in breaks:
            begin = self.timestamps[b-1]
            stamp = self.timestamps[b]-self.period
            replicate_row = data[b-1]
            
            #Fill the gap with appropriately placed timestamps and the replicate_row.
            while stamp > begin:
                timestamps.insert(b, copy.copy(stamp))
                imputed.insert(b, True)
                data = np.vstack((data[:b,:], replicate_row, data[b:,:]))
                stamp -= self.period
                
        timestamps = np.array(timestamps)
        imputed = np.array(imputed).squeeze()
        
        #Return a new time series representing the oversampled series.
        filled = {}
        filled['headers'] = self.headers
        filled['timestamps'] = timestamps
        filled['data'] = data
        filled['imputed']=imputed
        
        return Series(series=filled)
                
     
    def Downsample(self, factor):
        #Keep only every <factor>th observation.
        
        #Only works when the time series is complete (no breaks)
        if len(self.breaks) == 0:
        
            #Get working copies
            timestamps = copy.copy(self.timestamps)
            data = copy.copy(self.data)
            
            rows = range( len(timestamps) )
            rows = filter(lambda x: x % factor == 0, rows)
                    
            timestamps = timestamps[rows]
            data = data[rows]
            
            #Return a new time series representing the oversampled series.
            undersampled = {}
            undersampled['headers'] = self.headers
            undersampled['timestamps'] = timestamps
            undersampled['data'] = data
            
            return Series(series=undersampled)
            
        else: print "The time series has gaps, cannot undersample."
           
           
    def Upsample(self, factor):
        #Replicate each observation <factor> times.
        
        #Only works when the time series is complete (no breaks)
        if len(self.breaks) == 0:
        
            #Get working copies
            data = copy.copy(self.data)
            
            #Upsample the data rows
            factor = int(factor)
            rows = np.arange( len(self.timestamps) * factor )
            rows = rows // factor
            data = data[rows]
            
            #Linearly space the timestamps
            start = netcdftime.JulianDayFromDate(self.timestamps[0])
            end = netcdftime.JulianDayFromDate(self.timestamps[-1] + self.period)
            timestamps = np.linspace(start, end, num=len(rows)+1, endpoint=True)[:-factor]
            timestamps = np.array( map(netcdftime.DateFromJulianDay, timestamps) )
            
            
            
            #Return a new time series representing the oversampled series.
            oversampled = {}
            oversampled['headers'] = self.headers
            oversampled['timestamps'] = timestamps
            oversampled['data'] = data
            
            return Series(series=oversampled)
            
        else: print "The time series has gaps, cannot upsample."
        
        
    def Spin_Off(self, column):
        #Generate a new time series object from one of the columns in this one.
        
        #Get the column index
        index = self.Get_Index(column)
        
        #If we got an index out of the column parameter, then spin off the new time series
        try:
            spin_off = {}
            spin_off['headers'] = self.headers[index]
            spin_off['timestamps'] = self.timestamps
            spin_off['data'] = self.data[:,index]
            
            return Series(series=spin_off)
        
        #Otherwise, report the error
        except NameError:
            print "Cannot identify the specified column:",column
        
        
    def Differentiate(self, column):
        #Add a column to the time series that represents the difference between this measurement and the last.
        
        #Get the column index
        index = self.Get_Index(column)
        
        #If we got an index out of the column parameter, then add a new column to the time series.
        try:
            derivative = [np.nan] * self.data.shape[0]
            i=1
                
            while i < self.data.shape[0]:
                derivative[i]  = self.data[i,index] - self.data[i-1,index]
                i += 1
                
            self.headers.append("d" + self.headers[index] + "/dt")
            self.data = np.hstack( (self.data, np.array(derivative, ndmin=2).transpose()) )
        
        #Otherwise, report the error
        except NameError:
            print "Cannot identify the specified column:",column        
        
        
    '''def Impute(self, model=None, begin=None, end=None):
        #Fill gaps in the time series with imputed values
        
        self.imputed = np.zeros(len(self.timestamps), dtype=bool)
        
        imputations = []
        for c in range(self.data.shape[1]):
            imputations.append(np.mean(self.data[:,c]))
            
        imputations = np.array(imputations, ndmin=2)
        
        bb = copy.copy(self.breaks)
        bb.reverse()
        
        for b in bb:
            begin = self.timestamps[b-1]
            end = self.timestamps[b]
            
            impute = begin + 2*self.period
            insert_timestamps = [begin + self.period]
            insert_data = imputations
            flags = [True]
            
            while impute < end:
                insert_data = np.vstack( (insert_data, imputations) )
                insert_timestamps.append(impute)
                flags.append(True)
                impute += self.period
            
            self.data = np.vstack( (self.data[:b,:], insert_data, self.data[b:,:]) )
            self.timestamps = np.hstack( (self.timestamps[:b], insert_timestamps, self.timestamps[b:]) )
            self.imputed = np.hstack( (self.imputed[:b], flags, self.imputed[b:]) )'''
            
    
    def Get_Index(self, column):
        #Try to interpret the column parameter as an index
        try: index = int(column)
        
        #If that doesn't work, try to interpret it as the name of a column
        except ValueError:
        
            for header in self.headers:
                if column.lower() in header.lower():
                    index = self.headers.index(header)
                    
        return index
    
    
    def Wavelet(self, column, wavelet='db4'):
        #Decompose the time series using wavelets
        
        #Get the column index
        index = self.Get_Index(column)
        
        self.wavelet_coefs = pywt.wavedec(self.data[:,index], wavelet)
        
    
    def ARX_Model_Matrix(self, args):
        #Set up the X matrix of an ARX model to fit the data.
                
        #Get the model target's index
        index = self.Get_Index(args['target'])
        
        #If we got an index out of the column parameter, then generate the model
        if len(self.breaks) != 0:
            print "The time series has gaps, cannot make an ARX model."
        else:
            #Begin by setting up the intercept-only model.
            X = np.ones( shape=(self.data.shape[0],1) )
            
            #Eliminate the model's target from the list of drivers
            drivers = range(self.data.shape[1])
            drivers.remove(index)
            
            #Add the delayed, time-shifted inputs to the X matrix.                    
            for i in np.arange(len(args['nb'])):
                for j in np.arange(args['nb'][i])+1:
                    col = self.Lag(column=drivers[i], lag=j+args['nk'][i])
                    X = np.hstack( (X, col) )

            #Add the delayed outputs to the X matrix           
            for i in np.arange(args['na'])+1:
                col = self.Lag(column=index, lag=i)
                X = np.hstack( (X, col) )
                
            #Now we will strip out any imputed rows
            good_rows = mlab.find(self.imputed == False)
            X = X[good_rows,:]
             
        return X
        
    
    def ARX(self, args):
        #Create an ARX model with the parameters specified in args.
        
        index = self.Get_Index(args['target'])
        Y = self.data[:,index]
        X = self.ARX_Model_Matrix(args)
        args['matrix_generator'] = 'ARX_Model_Matrix'
        
        return Model(X, Y, args)
    
    
    def Lag(self, column, lag=1):
        #Add a new column to the time series, representing past values of an existing column.
        
        #Get the column index
        index = self.Get_Index(column)
        
        #Delay the measurement by <lag>
        lagged = [np.nan] * lag
        lagged.extend(self.data[0:-lag,index])
        lagged = np.array(lagged, ndmin=2).transpose() 
        
        #Return the lagged column
        return lagged
            
            
    def Predict(self, model):
        return model.Predict(self)
    
        
        


        
class Model:
    '''Represents an autoregression model'''
    
    def __init__(self, X, Y, args):
        #take the X, Y matrices and estimate coefficients by least squares.
        
        self.mask = mask = np.ones( X.shape[0], dtype=bool )
        
        #Remove rows with nans in the X matrix
        nan_rows = np.unique( mlab.find(np.isnan(X)) // X.shape[1] )
        mask[nan_rows] = False

        #Remove rows with nans in the Y vector
        nan_rows = mlab.find(np.isnan(Y))
        mask[nan_rows] = False


        self.X = X[mask]
        self.Y = Y[mask]
        
        #Use least squares to estimate the model parameters
        self.model = sm.OLS(self.Y,self.X)
        self.fit = self.model.fit()
        fitted = list(self.model.predict(self.X))
        residual = list(self.Y-self.model.predict(self.X))
        self.model_args = args
        
        #now put NaNs into the rows we removed so the output is the same shape as the input.
        nan_rows = list(mlab.find(mask==False))
        nan_rows.reverse()
        
        for row in nan_rows:
            fitted.insert(row, np.nan)
            residual.insert(row, np.nan)
        
        self.fitted = np.array(fitted).squeeze()
        self.residual = np.array(residual).squeeze()
        
        model_series = {}
        
        
        
    def Predict(self, series): 
        matrix_generator = getattr(series, self.model_args['matrix_generator'])
        X = matrix_generator(self.model_args)
        return self.model.predict(X)
        
        
        
class Source:
    def __init__(self, source):
    
        self.source = source
    
        if type(source) == dict:
            self.row = 0
            self.headers = source.keys()
            self.Next = self.__dict_next__
            
            
        elif type(source) == str:
            self.file = open(source, 'r')
            self.headers = self.file.readline().rstrip('\n').split(',')
            self.Next = self.__file_next__
            
            
    def __dict_next__(self):
        try: next = np.array(self.source.values()).transpose()[self.row]
        except IndexError: next = np.nan
        
        self.row += 1
        
        return next
        
        
    def __file_next__(self):
        next = self.file.readline()
        
        if not next: next = np.nan
        else: next = next.rstrip('\n').split(',')
        
        return next
        
        
        
        
        
        
        
        
        
        
        
        