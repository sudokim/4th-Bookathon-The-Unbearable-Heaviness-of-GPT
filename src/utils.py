from hashlib import md5


# import grpc


# from proto.message_log_pb2 import INFO
# from proto.message_log_pb2 import LogRequest
# from proto.message_log_pb2_grpc import MessageLoggerStub
#
# LOG_END_POINT = "localhost:36000"


# def request_log(msg: str, log_level: int = INFO) -> bool:
#     """
#     Request to log a message
#     :param msg: Message to log
#     :param log_level: Log level
#     :return: True if success, False otherwise
#     """
#     request = LogRequest(msg=msg, log_level=log_level)
#
#     with grpc.insecure_channel(LOG_END_POINT) as channel:
#         stub = MessageLoggerStub(channel)
#         response = stub.LogMessage(request)
#
#     return response


def hash_file_md5(path: str) -> str:
    """
    Calculate the MD5 hash of a file
    :param path: Path to the file
    :return: MD5 hex representation
    """
    h = md5()

    with open(path, "rb") as fp:
        chunk = 0
        while chunk != b"":
            chunk = fp.read(1024)
            h.update(chunk)

    return h.hexdigest()
