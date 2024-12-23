#----------------------------------------------------------------------------------#
# By Emanuel Nunez and Edward White
# Version 1.1s
#----------------------------------------------------------------------------------#
# Notes    
# For the code to work, the IP address must be set first. This can change everytime the computer is restarted.
# The IP address can be found by using the command `ipconfig` in the command prompt (Windows), or with `ifconfig` (Linux). 
# The port number is set to 8888 and is the same for the server and client.
# The private IP address for the computer is used and not the public one
#----------------------------------------------------------------------------------#
# Usage
# This code is used to connect to a server that is running on a computer. The server is listening for a connection from pc
# 1. Paste this code onto a Python box in Choregraphe
# 2. Change the output type of onStopped to string
# 3. Run the server code on the computer
# 4. Run the code on Choregraphe
#----------------------------------------------------------------------------------#


# Import the necessary modules (yes, even inside of choregraphe)
import socket
import random
import time

# Parameters for the server

host = '169.254.93.16'     # = 'FIND YOUR IP ADDRESS' # See above notes
# host = '172.0.0.1'          # Localhost, uncomment if running in simulation
port = 8888 # No need to change

client_socket = None


class MyClass(GeneratedClass):

    result = None

    def __init__(self):
        GeneratedClass.__init__(self)

    def onLoad(self):
        # Called when "play" is pressed
        pass


    def send_data(self, client_socket, data):
        self.logger.info("Sending data: %s",str(data))
        client_socket.sendall(str(data).encode())

    def onInput_onStart(self):

        ### 1. Define the server        
        self.logger.info("Connecting to socket...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect to the server
        self.logger.info("Connecting to server...")
        client_socket.connect((host, port))

        ### 2. Send a message
        self.send_data(client_socket,"poseCheck")
        # Small delay so the server can keep up
        time.sleep(0.001)
        # Sends a 1 to the server
        data = client_socket.recv(1024).decode()
        self.logger.info("Data recieved : %s", str(data))
        self.onStopped(str(data)) 
        client_socket.close()
        pass

    def onInput_onStop(self):
        self.onUnload() 
        self.onStopped(str(self.result)) #activate the output of the box

    def onUnload(self): 
        # Close the socket when stopped   
        if client_socket is not None:          
            client_socket.close()   
        pass