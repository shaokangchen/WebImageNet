import socket
import json
import myres

filename = './imagelist.txt'
q = myres.myquery(filename)
q.set_threshold(0.2)

host = socket.gethostname()    
port = 1003               # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
if True:    
    #string = q.convert_json()
    #string2 = json.dumps(q,default=q.serialize)
    string2 = json.dumps(q, cls=myres.ComplexEncoder, indent=4)
    #print string
    print string2
    s.sendall(string2)
    data = s.recv(1024)
    print('Received', repr(data))
    #res = myres.myquery.load_json(data)
    #res.display()
s.close()

