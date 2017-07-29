import socket
import time
import myres
import json
from multiprocessing import Pool, freeze_support


def main():
    host = ''        
    port = 1003    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(10)
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        print(elapsed_time)
        if elapsed_time > 10:
            s.close()    
            break
        conn, addr = s.accept()
        print('Connected by', addr)
        data = conn.recv(1024)
        if not data: continue
    
        search = myres.myquery.fromJSON(data)        
        search_res = myres.result_list()
        search_res.run_queries(search)
        string2 = json.dumps(search_res, cls = myres.ComplexEncoder, indent=4)
        print string2
        conn.sendall(search_res.conver_json())
        #conn.sendall(string2)
        conn.close()  
    


if __name__=="__main__":
    freeze_support()
    main()
    
    print('done')

