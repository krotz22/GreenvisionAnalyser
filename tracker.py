import math

class Tracker:
    def __init__(self, line_y):
        self.center_points = {}  # Store object centers
        self.id_count = 0  # Unique ID for objects
        self.counted_ids = set()  # Store counted object IDs
        self.line_y = line_y  # Y-position of the counting line

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2  # Center X
            cy = (y1 + y2) // 2  # Center Y

            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:  # Object is the same as previous frame
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, obj_id])

                    # If object crosses the line, count it once
                    if cy > self.line_y and obj_id not in self.counted_ids:
                        self.counted_ids.add(obj_id)

                    same_object_detected = True
                    break

            if not same_object_detected:  # New object detected
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        # Remove old IDs
        new_center_points = {obj_id: self.center_points[obj_id] for _, _, _, _, obj_id in objects_bbs_ids}
        self.center_points = new_center_points

        return objects_bbs_ids
