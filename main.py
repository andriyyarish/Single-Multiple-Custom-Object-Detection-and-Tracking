import test_data_reader
import object_tracker

parameters = test_data_reader.readParameters()

for p in parameters:
    object_tracker.start_processing(p)