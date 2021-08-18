class Parameters:
  def __init__(self, video_file, init_lines_image, bounding_rectangle_x_y):
    self.video_file = "./data/video/" + video_file.strip()
    self.output_file = "./data/video/output_" + video_file.replace(".mp4", ".avi")
    self.init_lines_image = "./data/video/" + init_lines_image.strip()
    x_y_array = bounding_rectangle_x_y.split("|")
    self.region_of_interest_rectangle = [(x_y_array[0], x_y_array[1]), (x_y_array[2], x_y_array[3]), (x_y_array[4], x_y_array[5]), (x_y_array[6], x_y_array[7])]

  def toString(self):
     print("Video: {}; init lines: {}".format(self.video_file, self.init_lines_image))

import csv

def readParameters() -> Parameters:
    with open('test_data.csv', newline='') as csvfile:
         reader = csv.reader(csvfile, delimiter=',')
         next(reader, None)  # skip the headers
         result = []
         for row in reader:
             p = Parameters(row[0], row[1], row[2])
             p.toString()
             result.append(p)
    return result



