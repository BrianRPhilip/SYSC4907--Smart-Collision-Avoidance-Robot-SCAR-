import io
import socket
import struct
import time
import picamera

pi_socket = socket.socket()

pi_socket.connect((socket.gethostbyname(socket.gethostname()), 8000))  # ADD Computer ip here HERE

print('Enter quit to stop sending images to server: ')

# Make a file-like object out of the connection. Object exposing a file-oriented API, in this case a buffer like object
connection = pi_socket.makefile('wb')
try:
    #Camera setup
    camera = picamera.PiCamera()
    camera.vflip = True
    camera.resolution = (500, 480)

    # Note the start time and construct a stream to hold image data
    # temporarily (we could write it directly to connection but in this
    # case we want to find out the size of each capture first to keep
    # our protocol simple)
    start = time.time()
    stream = io.BytesIO()
    while input() != quit:
        camera.capture(stream, 'jpeg')
        # Write the length of the capture to the stream and flush to
        # ensure it actually gets sent
        connection.write(struct.pack('<L', stream.tell()))
        connection.flush()
        # sets steam's current position
        stream.seek(0)
        connection.write(stream.read())
        stream.seek(0)
        stream.truncate()
        # need to optimize this sleep time depending on our algorithms efficiency
        time.sleep(.5)
    # Write a length of zero to the stream to signal we're done
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    pi_socket.close()
