import numpy as np


class DatasetShuffler(object):
    def __init__(self, input_array, epoch_count, batch_size):
        self.original_array = input_array
        self.epoch_count = epoch_count
        self.batch_size = batch_size
        self.current_array = np.ndarray([])
        self.current_epoch = 0
        self.current_array_empty = True

    def get_next(self):
        # print("get_next, shape: {}, size: {}".format(len(self.current_array.shape), self.current_array.size))
        # print("Length of shape: {}".format(len(self.current_array.shape)))
        # if len(self.current_array.shape) > 0:
        #     print("shape[0]: {}".format(self.current_array.shape[0]))

        # Enough data to return new batch
        if len(self.current_array.shape) > 0 and self.current_array.shape[0] > self.batch_size:
            self.batch = self.current_array[0:self.batch_size]
            self.current_array = np.delete(self.current_array, range(self.batch_size))
            return True

        # No more epoch for which we can get data
        if self.current_epoch >= self.epoch_count:
            return False

        self.current_epoch += 1
        self.current_array = np.copy(self.original_array)
        np.random.shuffle(self.current_array)
        self.batch = self.current_array[0:self.batch_size]
        self.current_array = np.delete(self.current_array, range(self.batch_size))
        return True

a = np.asarray([0,1,2,3,4,5,6,7,8,9])
# s = np.random.choice(a, len(a), replace=False)
# print(s)
# print(a)
# a = np.delete(a, range(2))
# print(a)

# print("Here we go")
# dss = DatasetShuffler(a, 2, 3)
# a = True
# while dss.get_next():
#     print(dss.batch)



