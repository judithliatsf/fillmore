import datetime

def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), s),
          flush=True)