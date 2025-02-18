import socket, os, sys, optparse, time

#output file-results
outfile = 'results.csv'
BufferSize = 1024

def retBanner(ip, port=22):
    sock = socket.socket()
    sock.connect((ip, port))
    sock.send(b'Gabbage')
    banner = sock.recv(1024)
    return banner


def download_file(targetHost, targetFile):
    connection_test = retBanner(targetHost)
    if connection_test is None:
        print('Port is closed')
        exit(0)
    else:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect((targetHost, 22))
        connection.send(bytes(targetFile, encoding='utf-8'))
        with open('./{}'.format(targetFile.split('/')[-1]), 'wb') as ff:
            while True:
                data = connection.recv(BufferSize)
                if not data:
                    break
                ff.write(data)
            ff.close()
        connection.close()
        print('Data Downloaded Succesful..!')

def main():
    parser = optparse.OptionParser('usage %prog -H <targetHost> -F <targetFile> -I <iterations>')
    parser.add_option('-H', dest='targetHost', type='string', help='Specify a connection port')
    parser.add_option('-F', dest='targetFile', type='string', help='Specify File omn remote to download')
    parser.add_option('-I', dest='iterations', type='int', help='Specify number of iterations')

    (options, args) = parser.parse_args()
    host = options.targetHost
    file = options.targetFile
    iterations = options.iterations

    headings = ["Iteration", "Throughput", "Time", "BufferSize"]
    data = []

    for iteration in range(iterations):
        start = time.time()
        download_file(targetHost=host, targetFile=file)
        end = time.time()

        lapse = (end - start)/1000
        with open('outfile', 'a') as ff:
            tp = round((BufferSize*0.001) / (lapse+0.000001), 3)
            smallist = [iteration, tp, lapse, BufferSize]
            data.append(smallist)
            ff.write('iteration {},{},{},{}\n'.format(iteration, tp, lapse, BufferSize))
        ff.close()
    format_row = "{:>12}" * (len(headings) + 1)
    print(format_row.format("", *headings))
    for row in data: 
        print(format_row.format('', *row))
    import subprocess

    file_ = open('res.txt', 'w+')
    subprocess.run('traceroute google.com', shell=True, stdout=file_)
    file_.close()

    with open('res.txt', 'r') as f:
            print('hellotrace',f.read())
       

if __name__ == '__main__':
    main()
