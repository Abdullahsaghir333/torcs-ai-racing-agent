import sys
import argparse
import socket
import driver
import carControl

if __name__ == '__main__':
    pass

# === Arg Parser ===
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
parser.add_argument('--host', dest='host_ip', default='localhost')
parser.add_argument('--port', dest='host_port', type=int, default=3001)
parser.add_argument('--id', dest='id', default='SCR')
parser.add_argument('--maxEpisodes', dest='max_episodes', type=int, default=1)
parser.add_argument('--maxSteps', dest='max_steps', type=int, default=0)
parser.add_argument('--track', dest='track', default=None)
parser.add_argument('--stage', dest='stage', type=int, default=3)
args = parser.parse_args()

# === Connect Summary ===
print('Connecting to server host ip:', args.host_ip, '@ port:', args.host_port)
print('Bot ID:', args.id)
print('Maximum episodes:', args.max_episodes)
print('Maximum steps:', args.max_steps)
print('Track:', args.track)
print('Stage:', args.stage)
print('*********************************************')

# === Create Socket ===
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
except socket.error:
    print('Could not create socket.')
    sys.exit(-1)

shutdownClient = False
curEpisode = 0
verbose = False

# === Initialize AI Driver ===
d = driver.Driver(args.stage)

while not shutdownClient:
    while True:
        print('Sending id to server: ', args.id)
        buf = args.id + d.init()
        print('Sending init string to server:', buf)

        try:
            sock.sendto(buf.encode(), (args.host_ip, args.host_port))
            buf, _ = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error:
            print("No response from server...")
            continue

        if '***identified***' in buf:
            print('Server accepted connection.')
            break

    currentStep = 0

    while True:
        try:
            buf, _ = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error:
            continue

        if buf.startswith('***shutdown***'):
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break

        if buf.startswith('***restart***'):
            d.onRestart()
            print('Client Restart')
            break

        currentStep += 1
        if currentStep != args.max_steps:
            if buf:
                response = d.drive(buf)
        else:
            response = '(meta 1)'

        if verbose:
            print('Sending:', response)

        if response:
            try:
                sock.sendto(response.encode(), (args.host_ip, args.host_port))
            except socket.error:
                print("Failed to send data... Exiting.")
                sys.exit(-1)

    curEpisode += 1
    if curEpisode == args.max_episodes:
        shutdownClient = True

sock.close()
