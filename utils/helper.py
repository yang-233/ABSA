import time
import requests
import functools
import json
import os
import dill
from datetime import datetime
from IPython.lib import kernel
from tensorboardX import SummaryWriter

def get_name_of_notebook():
    connection_file_path = kernel.get_connection_file()
    connection_file = os.path.basename(connection_file_path)
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]
    s = requests.Session()
    s.trust_env = False
    c = {"_xsrf" : "2|023f1ddd|860b2d16d4921068cc4f21744b01f870|1574233833",
        "username-172-21-15-34-8888" :"2|1:0|10:1577086728|26:username-172-21-15-34-8888|44:M2NlZWE5YzkwOGJmNDY3MjgxZWZmNzkyODBkOTVhOTU=|363cfe162cbf74d3a9e912bf119f26414f5d0f0102573a85a7457202787da51d"
        }
    response = s.get('http://172.21.15.34:8888/api/sessions?', 
                      cookies=c)
    sessions = json.loads(response.text)
    for sess in sessions:
        if sess['kernel']['id'] == kernel_id:
            return sess['notebook']['name'].split('.')[0]

def dump_notebook():
    dill.dump_session(get_name_of_notebook() + ".session")

def load_notebook():
    session_name = get_name_of_notebook() + ".session"
    if session_name in os.listdir(os.getcwd()):
        dill.load_session(session_name)

def notice(to='1281825023@qq.com'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            func_name = func.__name__
            t0 = time.time()
            res = func(*args, **kw)
            t1 = time.time()
            start_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                   time.localtime(t0))
            spend_time = int(t1 - t0)
            hours = spend_time // 3600
            spend_time %= 3600
        
            minutes = spend_time // 60
            spend_time %= 60
        
            seconds = spend_time
        
            spend_time = f'{hours}h {minutes}m {seconds}s'
        
            content = 'Function ' + func_name + ' has done:\n' + \
                  '    Start at ' + start_time + '\n' + \
                  '    Spend ' + spend_time + '\n' +  \
                  '    result:\n' + res
            
            r = requests.post('http://104.168.211.229:5000/sendEmail', data=
                         {
                             "to" : to,
                             "title" : "The program has been completed!",
                             "content" : content
                         })
            print(r.text)

            return res
        return wrapper
    return decorator

def notice_run(func, args:tuple, to='1281825023@qq.com'):
    func_name = func.__name__
    t0 = time.time()
    res = func(*args)
    t1 = time.time()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))
    spend_time = int(t1 - t0)
    hours = spend_time // 3600
    spend_time %= 3600
    minutes = spend_time // 60
    spend_time %= 60
    seconds = spend_time
    spend_time = f'{hours}h {minutes}m {seconds}s'
        
    content = 'Function ' + func_name + ' has done:\n' + \
                  '    Start at ' + start_time + '\n' + \
                  '    Spend ' + spend_time + '\n' +  \
                  '    result:\n' + res
    
    r = requests.post('http://104.168.211.229:5000/sendEmail', data=
                        {
                             "to" : to,
                             "title" : "The program has been completed!",
                             "content" : content
                         })        
    print(r.text)
    
def time_writer():
    TIMESTAMP = "{0:%Y-%m-%d %H:%M:%S/}".format(datetime.now())
    return SummaryWriter('log/' + TIMESTAMP)