import socket
import json
import myres


#q = myquery.myquery()
#print q.get_count()
filename = './imagelist.txt'
#q = myquery.myquery(filename)
#q.getlist_fromfile(filename)
q = myres.myquery(filename)

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

