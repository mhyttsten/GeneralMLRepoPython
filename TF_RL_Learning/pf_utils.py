import tensorflow as tf
import time

debug_flag = True

def sleep(message, time_in_s):
    global debug_flag
    if debug_flag:
        time.sleep(1)
        print(message)
        time.sleep(time_in_s)

def debug(message):
    global debug_flag
    if debug_flag:
        print(message)

def debug_tensor(tensor, message):
    global debug_flag
    if debug_flag:
        tensor = tf.Print(tensor, [tensor], message=message, summarize=100)
    return tensor

def debug_tensor_shape(tensor, message):
    global debug_flag
    if debug_flag:
        tensor = tf.Print(tensor, [tf.shape(tensor)], message=message, summarize=100)
    return tensor
