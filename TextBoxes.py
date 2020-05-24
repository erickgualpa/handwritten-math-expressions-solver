import numpy as np
import matplotlib.path as mplPath

START_X = 0
START_Y = 1
END_X = 2
END_Y = 3

class TextBoxes:
    def __init__(self, boxes, min_dist_allowed):
        self.__boxes = boxes
        self.__boxes_hierarchy = {}
        self.__min_dist_allowed = min_dist_allowed
        self.build_boxes_hierarchy()

    def check_overlapping_between_two_boxes(self, box, neighbor):   # 'box' and 'neighbor' have this form:
                                                                    # (top_left, top_right, bottom_left, bottom_right)
        overlapping = False

        ## Neighbor points #################################################
        neighbor_top_left = (neighbor[0][0], neighbor[0][1])
        neighbor_bottom_left = (neighbor[2][0], neighbor[2][1])
        ####################################################################

        ## Main box vertices ###############################################
        box_top_left = [box[0][0], box[0][0]]
        box_top_right = [box[1][0], box[1][1]]
        box_bottom_left = [box[2][0], box[2][1]]
        box_bottom_right = [box[3][0], box[3][1]]
        ####################################################################

        ## Build bounding box ##############################################
        boundingBox = mplPath.Path(np.array([box_top_left,
                                       box_top_right,
                                       box_bottom_left,
                                       box_bottom_right]))
        ####################################################################

        if boundingBox.contains_point(neighbor_top_left) or boundingBox.contains_point(neighbor_bottom_left):
            overlapping = True
        return overlapping

    def euclidean_distance_between_two_boxes(self, box, neighbor):
        # Working with the box1 right-bottom vertex and the box2 left-bottom vertex

        ## Neighbor bottom left vertex ######################################
        neighbor_point = np.array([neighbor[2][0], neighbor[2][1]])
        ####################################################################

        ## Main box bottom right vertex ####################################
        main_box_point = np.array([box[3][0], box[3][1]])
        ####################################################################

        ## Calculate Euclidean distance ####################################
        dist = (main_box_point - neighbor_point)**2
        dist = np.sum(dist, axis=1)
        return np.sqrt(dist)
        ####################################################################

    def build_boxes_hierarchy(self):
        box_index = 0
        for box in self.__boxes:
            # Compute box vertices
            top_left = (box[START_X], box[START_Y])     # (startX, startY)
            top_right = (box[END_X], box[START_Y])      # (endX, startY)
            bottom_left = (box[START_X], box[END_Y])    # (startX, endY)
            bottom_right = (box[END_X], box[END_Y])     # (endX, endY)

            self.__boxes_hierarchy['box' + str(box_index)] = {}
            self.__boxes_hierarchy['box' + str(box_index)]['vertices'] = (top_left, top_right, bottom_left, bottom_right)
            box_index += 1

        for box_id in self.__boxes_hierarchy:
            self.__boxes_hierarchy[box_id]['neighbors'] = []
            self.__boxes_hierarchy[box_id]['overlapped'] = []
            self.__boxes_hierarchy[box_id]['near'] = []
            for neighbor_box_id in self.__boxes_hierarchy:
                if neighbor_box_id != box_id:
                    current_box = self.__boxes_hierarchy[box_id]['vertices']
                    neighbor_box = self.__boxes_hierarchy[neighbor_box_id]['vertices']

                    # Check overlapping
                    if self.check_overlapping_between_two_boxes(current_box, neighbor_box):
                        self.__boxes_hierarchy[box_id]['overlapped'].append(neighbor_box_id)

                    # Check distance between boxes
                    if self.euclidean_distance_between_two_boxes(current_box, neighbor_box) < self.__min_dist_allowed:
                        self.__boxes_hierarchy[box_id]['near'].append(neighbor_box_id)

        return self.__boxes_hierarchy