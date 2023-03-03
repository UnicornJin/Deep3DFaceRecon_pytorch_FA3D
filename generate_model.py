'''
This Script tests the model generation through socket commands
'''

import sys, getopt
import socket

HELP_STRING = 'This is the script to generate models.\ngenerate_model.py -i <inputfile> -o <outputfile>\n'
MODEL_PORT = 11451

def gen_model(filename):
    s = socket.socket()
    host = socket.gethostname()
    port = MODEL_PORT

    s.connect((host, port))
    print("[Runner] Model connected")

    # folder = './userimages/' + filename.replace('.jpg', '') + '/'
    folder = './userimages/' + filename + '/'
    print("[Runner] Folder to generate: ", folder)
    s.sendall(folder.encode())
    print("[Runner] Image info sent")

    print("[Runner] Model Generation Finished", s.recv(1024).decode())

    s.close()

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print(HELP_STRING)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(HELP_STRING)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    gen_model(inputfile)


if __name__ == "__main__":
    main(sys.argv[1:])