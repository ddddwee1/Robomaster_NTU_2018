import socket

NH_SERVER = 0
NH_CLIENT = 1

class NetworkHandler:
    def __init__(self, type, ):
        if type == NH_SERVER:

            self._soc = socket(AF_INET, SOCK_STREAM)
            self._soc.bind((HostIP, Port))
            self._soc.listen(100)
