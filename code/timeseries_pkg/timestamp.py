from dateutil import parser
import datetime
from calendar import isleap
import numpy as np


class Stamper:

    def __init__(self, headers, **args):
        self.time_columns = [] # This will be a list of the columns used to make the timestamp.
        (date_parser, time_parser) = self.Select_Parsers(headers)
        self.Parse = self.Build_Parser(date_parser, time_parser, headers, **args)
        

    def Select_Parsers(self, heads):
        date_parser = 0 # We'll set date_parser to identify the date parsing routine
        time_parser = 0 # We'll set time_parser to identify the time parsing routine
                        
                
        if filter( lambda x: 'datetime' in x.lower(), heads):
            date_parser = 1
        
        else:
            if filter( lambda x: 'date' in x.lower(), heads):
                date_parser = 2
                if filter( lambda x: 'time' in x.lower(), heads): time_parser = 1
                elif filter( lambda x: 'hour' in x.lower(), heads):
                    if filter( lambda x: 'minute' in x.lower(), heads):
                        if filter( lambda x: 'second' in x.lower(), heads): time_parser = 2
                        else: time_parser = 7
                    elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 3
                    else: time_parser = 6
                elif filter( lambda x: 'minute' in x.lower(), heads):
                    if filter( lambda x: 'second' in x.lower(), heads): time_parser = 4
                    else: time_parser = 8
                elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 5
                
            elif filter( lambda x: 'year' in x.lower(), heads):
                if ( filter( lambda x: 'doy' in x.lower(), heads) or filter( lambda x: 'julian' in x.lower(), heads) ):
                    date_parser = 3
                    if filter( lambda x: 'time' in x.lower(), heads): time_parser = 1
                    elif filter( lambda x: 'hour' in x.lower(), heads):
                        if filter( lambda x: 'minute' in x.lower(), heads):
                            if filter( lambda x: 'second' in x.lower(), heads): time_parser = 2
                            else: time_parser = 7
                        elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 3
                        else: time_parser = 6
                    elif filter( lambda x: 'minute' in x.lower(), heads):
                        if filter( lambda x: 'second' in x.lower(), heads): time_parser = 4
                        else: time_parser = 8
                    elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 5
    
                elif filter( lambda x: 'month' in x.lower(), heads):
                    if filter( lambda x: 'day' in x.lower(), heads):
                        date_parser = 4
                        if filter( lambda x: 'time' in x.lower(), heads): time_parser = 1
                        elif filter( lambda x: 'hour' in x.lower(), heads):
                            if filter( lambda x: 'minute' in x.lower(), heads):
                                if filter( lambda x: 'second' in x.lower(), heads): time_parser = 2
                                else: time_parser = 7
                            elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 3
                            else: time_parser = 6
                        elif filter( lambda x: 'minute' in x.lower(), heads):
                            if filter( lambda x: 'second' in x.lower(), heads): time_parser = 4
                            else: time_parser = 8
                        elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 5
                        
                    else: # Month but no day
                        date_parser = 5
                        if filter( lambda x: 'time' in x.lower(), heads): time_parser = 1
                        elif filter( lambda x: 'hour' in x.lower(), heads):
                            if filter( lambda x: 'minute' in x.lower(), heads):
                                if filter( lambda x: 'second' in x.lower(), heads): time_parser = 2
                                else: time_parser = 7
                            elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 3
                            else: time_parser = 6
                        elif filter( lambda x: 'minute' in x.lower(), heads):
                            if filter( lambda x: 'second' in x.lower(), heads): time_parser = 4
                            else: time_parser = 8
                        elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 5
                        
                elif filter( lambda x: 'day' in x.lower(), heads): #Day, but no month
                    date_parser = 3
                    if filter( lambda x: 'time' in x.lower(), heads): time_parser = 1
                    elif filter( lambda x: 'hour' in x.lower(), heads):
                        if filter( lambda x: 'minute' in x.lower(), heads):
                            if filter( lambda x: 'second' in x.lower(), heads): time_parser = 2
                            else: time_parser = 7
                        elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 3
                        else: time_parser = 6
                    elif filter( lambda x: 'minute' in x.lower(), heads):
                        if filter( lambda x: 'second' in x.lower(), heads): time_parser = 4
                        else: time_parser = 8
                    elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 5
                
                else: #Fractional year, presumably
                    date_parser = 6
                    if filter( lambda x: 'time' in x.lower(), heads): time_parser = 1
                    elif filter( lambda x: 'hour' in x.lower(), heads):
                        if filter( lambda x: 'minute' in x.lower(), heads):
                            if filter( lambda x: 'second' in x.lower(), heads): time_parser = 2
                            else: time_parser = 7
                        elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 3
                        else: time_parser = 6
                    elif filter( lambda x: 'minute' in x.lower(), heads):
                        if filter( lambda x: 'second' in x.lower(), heads): time_parser = 4
                        else: time_parser = 8
                    elif filter( lambda x: 'second' in x.lower(), heads): time_parser = 5
        
        
        return (date_parser, time_parser)
        
    
    def Build_Parser(self, date_parser, time_parser, headers, **args):
        '''Builds a function that will put a datetime stamp on a row of data'''
        
        if date_parser == 1:
            datetime_column = filter( lambda x: 'datetime' in x.lower(), headers )[0]
            datetime_column = headers.index(datetime_column)
            self.time_columns.append(datetime_column)
            
            def Parser(data_row):
                return parser.parse(data_row[datetime_column], **args)  
                
        else:        
            if date_parser == 2:
                date_column = filter( lambda x: 'date' in x.lower(), headers )[0]
                date_column = headers.index(date_column)
                def Date_Parse(data_row, **args):
                    return parser.parse(data_row[date_column])
                    
                    
            elif date_parser == 3:
                julian_column = (filter( lambda x: 'doy' in x.lower(), headers) or filter( lambda x: 'julian' in x.lower(), headers) or filter( lambda x: 'day' in x.lower(), headers))[0]
                year_column = filter( lambda x: 'year' in x.lower(), headers )[0]
                
                julian_column = headers.index(julian_column)
                year_column = headers.index(year_column)
                
                self.time_columns.append(julian_column)
                self.time_columns.append(year_column)
                
                def Date_Parse(data_row, **args):
                    days = int(data_row[julian_column])
                    days = datetime.timedelta(days=days-1)
                    years = int(data_row[year_column])
                    
                    return datetime.datetime(day=1, month=1, year=years) + days
            
            
            elif date_parser == 4:
                year_column = filter( lambda x: 'year' in x.lower(), headers )[0]
                month_column = filter( lambda x: 'month' in x.lower(), headers )[0]
                day_column = filter( lambda x: 'day' in x.lower(), headers )[0]
                
                year_column = headers.index(year_column)
                month_column = headers.index(month_column)
                day_column = headers.index(day_column)
                
                self.time_columns.append(day_column)
                self.time_columns.append(month_column)
                self.time_columns.append(year_column)
                
                def Date_Parse(data_row, **args):
                    days = int(data_row[day_column])
                    months = int(data_row[month_column])
                    years = int(data_row[year_column])
                
                    return datetime.datetime(day=days, month=months, year=years)
     
     
            elif date_parser == 5: #I do not expect to ever see date in fractional-month format
                pass
            
            
            elif date_parser == 6:
                year_column = filter( lambda x: 'year' in x.lower(), headers )[0]
                year_column = headers.index(year_column)
                
                self.time_columns.append(year_column)
                
                def Date_Parse(data_row, **args):
                    whole_year = int(data_row[year_column])
                    fractional_year = float(data_row[year_column]) % 1
                    
                    if isleap(whole_year):
                        days = int(fractional_year * 366)
                    else: days = int(fractional_year * 365)
                        
                    days = datetime.timedelta(days=days-1)
                    
                    return datetime.datetime(day=1, month=1, year=whole_year) + days

            
    
    
                    
                    
            if time_parser == 1:
                time_column = filter( lambda x: 'time' in x.lower(), headers )[0]
                time_column = headers.index(time_column)
                
                self.time_columns.append(time_column)
                
                def Time_Parse(data_row, **args):
                    timestamp = parser.parse(data_row[time_column]).time()
                    return datetime.timedelta(hours=timestamp.hour, minutes=timestamp.minute, seconds=timestamp.second)
            
            
            elif time_parser == 2:
                hour_column = filter( lambda x: 'hour' in x.lower(), headers )[0]
                minute_column = filter( lambda x: 'minute' in x.lower(), headers )[0]
                second_column = filter( lambda x: 'second' in x.lower(), headers )[0]
                
                hour_column = headers.index(hour_column)
                minute_column = headers.index(minute_column)
                second_column = headers.index(second_column)
                
                self.time_columns.append(hour_column)
                self.time_columns.append(minute_column)
                self.time_columns.append(second_column)
                
                def Time_Parse(data_row, **args):
                    hours = int(data_row[hour_column])
                    minutes = int(data_row[minute_column])
                    seconds = int(data_row[second_column])
                    
                    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
     
     
            elif time_parser == 3: #I do not expect to ever see time in hours-and-seconds format, but...
                hour_column = filter( lambda x: 'hour' in x.lower(), headers )[0]
                second_column = filter( lambda x: 'second' in x.lower(), headers )[0]
                
                hour_column = headers.index(hour_column)
                second_column = headers.index(second_column)
                
                self.time_columns.append(hour_column)
                self.time_columns.append(second_column)
                
                def Time_Parse(data_row, **args):
                    seconds = int(data_row[second_column])
                    minutes = int(seconds // 60)
                    seconds = int(seconds % 60)
                    hours = int(data_row[hour_column])
                                        
                    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
            
            elif time_parser == 4: #Minutes-and-seconds format
                minute_column = filter( lambda x: 'minute' in x.lower(), headers )[0]
                second_column = filter( lambda x: 'second' in x.lower(), headers )[0]
                
                minute_column = headers.index(minute_column)
                second_column = headers.index(second_column)
                
                self.time_columns.append(minute_column)
                self.time_columns.append(second_column)
                
                def Time_Parse(data_row, **args):
                    minutes = float(data_row[minute_column])
                    hours = int(minutes // 60)
                    minutes = int(minutes % 60)
                    seconds = int(data_row[second_column])
                    
                    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
                
            
            elif time_parser == 5: #Seconds only
                second_column = filter( lambda x: 'second' in x.lower(), headers )[0]
                second_column = headers.index(second_column)
                
                self.time_columns.append(second_column)
                
                def Time_Parse(data_row, **args):
                    seconds = float(data_row[second_column])
                    hours = int(seconds // 3600)
                    minutes = int(seconds // 60)
                    seconds = int(seconds % 60)
                    
                    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
     
            
            elif time_parser == 6: #Fractional hours only
                hour_column = filter( lambda x: 'hour' in x.lower(), headers )[0]
                hour_column = headers.index(hour_column)
                
                self.time_columns.append(hour_column)
                
                def Time_Parse(data_row, **args):
                    hours = float(data_row[hour_column])
                    whole_hours = int(hours)
                    fractional_hours = hours % 1
                    
                    minutes = fractional_hours * 60
                    seconds = int(minutes % 1 * 60)
                    minutes = int(minutes)
                    
                    return datetime.timedelta(hours=whole_hours, minutes=minutes, seconds=seconds)
      
            
            elif time_parser == 7: #hours and minutes
                hour_column = filter( lambda x: 'hour' in x.lower(), headers )[0]
                minute_column = filter( lambda x: 'minute' in x.lower(), headers )[0]
                
                hour_column = headers.index(hour_column)
                minute_column = headers.index(minute_column)
                
                self.time_columns.append(hour_column)
                self.time_columns.append(minute_column)
                
                def Time_Parse(data_row, **args):
                    hours = int(data_row[hour_column])
                    minutes = float(data_row[minute_column])
                    seconds = int(minutes % 1 * 60)
                    minutes = int(minutes)
                    
                    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
            
            
            elif time_parser == 8: #minutes only
                minute_column = filter( lambda x: 'minute' in x.lower(), headers )[0]
                minute_column = headers.index(minute_column)
                
                self.time_columns.append(minute_column)
                
                def Time_Parse(data_row, **args):
                    minutes = float(data_row[minute_column])
                    
                    seconds = int(minutes % 1 * 60)
                    hours = int(minutes // 60)
                    minutes = int(minutes % 60)
                    
                    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
                    
                    
            elif date_parser == 3: #Fractional day
                julian_column = (filter( lambda x: 'doy' in x.lower(), headers) or filter( lambda x: 'julian' in x.lower(), headers) or filter( lambda x: 'day' in x.lower(), headers))[0]
                julian_column = headers.index(julian_column)

                #julian_column is already added to self.time_columns.
                
                def Time_Parse(data_row, **args):
                    days = float(data_row[julian_column])
                    day_fraction = days % 1 
                    hours = int(day_fraction*24)
                    minutes = int(day_fraction*60*24 - (hours*60))
                    seconds = int(day_fraction*60*60*24 - (hours*60*60) - (minutes*60)) 
                    
                    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
                    
            else: #We got nothing
                def Time_Parse(data_row, **args):
                    return datetime.timedelta(hour=0, minute=0, second=0)
            
            def Parser(data_row, **args):
                date = Date_Parse(data_row, **args)
                time = Time_Parse(data_row, **args)
                
                
                return date+time
                
        return Parser
                
        
        
        
        
        
        
        
        