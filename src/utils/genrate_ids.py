import base64
import os

def genrate_id():
    return base64.b64encode(os.urandom(32))[:8]