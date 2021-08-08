import requests
from geotools.constants import *
from geotools._exceptions import *


class Alert:
    @staticmethod
    def alert_server(func):
        def wrapper():
            # call the function of the decorator
            res, filename = func()
            # default on fail message to be
            onfail_message = 'Alert message failed as ' + \
                str(func).split()[1] + '() didn\'t return a success message'

            # check whether the function of the decorator is success
            if res:
                print("Success: Stored in path")
                requests.get(f'http://localhost:{PORT}//alert/{filename}')
            else:
                raise PredictionError(onfail_message)
        return wrapper
