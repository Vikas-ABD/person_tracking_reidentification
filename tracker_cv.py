import cv2
import numpy as np
import math

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Store the appearance (template) of the last detected person
        self.last_person_template = None
        # Threshold for re-identification
        self.similarity_threshold = 0.7
        # Keep the count of the IDs
        # Each time a new object id is detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect, frame):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected, we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Re-identification logic using template matching
        if self.last_person_template is not None:
            new_templates = []

            for obj_bb_id in objects_bbs_ids:
                x, y, w, h, object_id = obj_bb_id
                person_roi = frame[y:y+h, x:x+w]
                person_template = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

                # Compare the appearance of the current person with the last detected person
                result = cv2.matchTemplate(person_template, self.last_person_template, cv2.TM_CCOEFF_NORMED)
                _, similarity = cv2.minMaxLoc(result)

                if similarity > self.similarity_threshold:
                    # Assign the same ID as the last person
                    obj_bb_id[4] = self.id_count - 1

                new_templates.append(person_template)

            # Update the last person template
            if new_templates:
                self.last_person_template = np.mean(new_templates, axis=0).astype(np.uint8)

        return objects_bbs_ids

