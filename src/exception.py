import sys 
import logging

def error_fun(error,error_details:sys):

    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in py script {0} in line number {1} error message {2}".format(file_name,exc_tb.tb_lineno,str(error))

    return error_message

class CustomeException(Exception):
    def __init__(self,error_message,error_details:sys):

        super().__init__(error_message)
        self.error_message=error_fun(error_message,error_details=error_details)
    
    def __str__(self):
        return self.error_message
    


