from shapely.geometry import Polygon
import math


def collision_inspect(x1, y1, s1_size, x2, y2, s2_size):
    # Calculate collision under the assumption that
    # size & coordinate of the goal & obstacles are known to us
    poly1 = Polygon([(x1, y1),
                     (x1 + s1_size, y1),
                     (x1 + s1_size, y1 + s1_size),
                     (x1, y1 + s1_size)])
    poly2 = Polygon([(x2, y2),
                     (x2 + s2_size, y2),
                     (x2 + s2_size, y2 + s2_size),
                     (x2, y2 + s2_size)])
    return poly1.intersects(poly2)


def gazebo_collision_inspect(x1, y1, radius1, x2, y2, radius2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if distance <= radius1 + radius2:
        return True
    else:
        return False


def crazyflie_collision_inspect(element, x2, y2, z2, radius2=0.0225):
    # radius_rotor: 0.0225 m 
    # We set the radius of the plane as the length of its longest part.
    if len(element) == 5:
        x1, y1, z1, r1, h1 = element
        # Code for the cylinder
        ball_center = (x2, y2, z2)
        cylinder_center = (x1, y1, z1)

        # Set the radius of the ball and the cylinder
        ball_radius = radius2
        cylinder_radius = r1
        cylinder_height = h1

        # Calculate the distance between the centers of the ball and cylinder on xy plane
        xy_distance = math.sqrt((ball_center[0] - cylinder_center[0]) ** 2 + (ball_center[1] - cylinder_center[1]) ** 2)

        # Check if the distance is less than the sum of the radii
        if xy_distance < ball_radius + cylinder_radius:
            # Check if the height of the ball is within the height of the cylinder
            if abs(ball_center[2] - cylinder_center[2]) < cylinder_height / 2 + ball_radius:
                # print("cyl crashed: {}, ({}, {}, {})\n".format(element, x2, y2, z2))
                return True
            else:
                return False
        else:
            return False
    elif len(element) == 6:
        x1, y1, z1, w1, l1, h1 = element
        ball_x, ball_y, ball_z, ball_radius = x2, y2, z2, radius2
        cub_x1 = x1 - w1 / 2
        cub_y1 = y1 - l1 / 2
        cub_z1 = z1 - h1 / 2
        cub_x2 = x1 + w1 / 2
        cub_y2 = y1 + l1 / 2
        cub_z2 = y1 + h1 / 2
        dx = abs(ball_x - max(cub_x1, min(ball_x, cub_x2)))
        dy = abs(ball_y - max(cub_y1, min(ball_y, cub_y2)))
        dz = abs(ball_z - max(cub_z1, min(ball_z, cub_z2)))

        global_distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if global_distance <= ball_radius:
            # print("global_dis: {} | ball: {} | box crashed: {}, ({}, {}, {})\n".format(global_distance, ball_radius, element, x2, y2, z2))
            return True
        else:
            return False

    # elif len(element) == 6:
    #     x1, y1, z1, w1, l1, h1 = element
    #     #Code for the cuboid
    #     # Set the center coordinates of the ball
    #     ball_center = (x2, y2, z2)
    #     ball_radius = radius2
    #     # Set the center coordinate, length, width and height of the cuboid
    #     cuboid_center = (x1, y1, z1)
    #     cuboid_length = l1
    #     cuboid_width = w1
    #     cuboid_height = h1
    #     cx = max(cuboid_center[0] - cuboid_width / 2,
    #                            min(ball_center[0], cuboid_center[0] + cuboid_width / 2))
    #     cy = max(cuboid_center[1] - cuboid_length / 2,
    #                            min(ball_center[1], cuboid_center[1] + cuboid_length / 2))
    #     cz = max(cuboid_center[2] - cuboid_height / 2,
    #                            min(ball_center[2], cuboid_center[2] + cuboid_height / 2))
    #     closest_point = [cx, cy, cz]
    #     global_distance = math.sqrt((closest_point[0] - ball_center[0]) ** 2 + (closest_point[1] - ball_center[1]) ** 2 + (closest_point[2] - ball_center[2]) ** 2)
    #     print("obs:{} | closest_point: {} | ball:{} |global_dis:{}".format(element, closest_point, ball_center, global_distance))
    #     if global_distance <= ball_radius:
    #         print("box crashed: {}, ({}, {}, {})\n".format(element, x2, y2, z2))
    #         return True
    #     else:
    #         return False


def grid_world_inspect(element, x, y, grid_step):
    # x, y, grid_step refers to the grid center
    # grid is all square
    # grid_step is just the half_box size
    h = grid_step
    if len(element) == 3:
        # obstacle is cylinder
        xo, yo, ro = element
        rect_x1 = x - h
        rect_y1 = y - h
        rect_x2 = x + h
        rect_y2 = y + h
        # Find the distance between center of circle and closest point on rectangle
        dx = abs(xo - max(rect_x1, min(xo, rect_x2)))
        dy = abs(yo - max(rect_y1, min(yo, rect_y2)))
        if dx ** 2 + dy ** 2 <= ro ** 2:
            return True
        else:
            return False
    elif len(element) == 4:
        x1, y1, w1, h1 = element
        poly1 = Polygon([(x1 - w1/2, y1 - h1/2),
                         (x1 + w1/2, y1 - h1/2),
                         (x1 + w1/2, y1 + h1/2),
                         (x1 - w1/2, y1 + h1/2)])
        poly2 = Polygon([(x - h, y - h),
                         (x + h, y - h),
                         (x + h, y + h),
                         (x - h, y + h)])
        return poly1.intersects(poly2)
    else:
        print("Non-cylinder obstacles (except walls) not accepted yet")
        raise NotImplementedError


def all_collision_inspect(element, x2, y2, radius2=0.2078):
    # The turtlebot has width of 0.306 (adding 2 wheels); length/height of 0.281
    # x2 = x2 - 0.064  # offset (-0.064) 
    if len(element) == 3:
        # cylinder (circle in 2d) inspection
        x1, y1, radius1 = element
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if distance < radius1 + radius2:
            # print(element, x2, y2, radius2)
            return True
        else:
            return False

    elif len(element) == 4:
        # brutal method
        x1, y1, _, _ = element
        if x1 > 0 and x2 + radius2 >= x1 - 0.01:
            # right wall
            return True
        elif x1 < 0 and x2 - radius2 <= x1 + 0.01:
            # left wall
            return True
        elif y1 > 0 and y2 + radius2 >= y1 - 0.01:
            # print("hit up wall")
            # up wall
            return True
        elif y1 < 0 and y2 - radius2 <= y1 - 0.01:
            return True 
        else:
            return False
            
 
    
    # elif len(element) == 4:
    #     # line segment inspection
    #     x1, y1, w1, h1 = element
    #     line_start_x, line_start_y, line_end_x, line_end_y = 0, 0, 0, 0
    #     if w1 >= h1:
    #         line_start_x, line_start_y = x1 - w1, y1
    #         line_end_x, line_end_y = x1 + w1, y1

    #     else:
    #         line_start_x, line_start_y = x1, y1 + h1
    #         line_end_x, line_end_y = x1, y1 - h1
    #     distance = abs(
    #         ((line_end_y - line_start_y) * x2) - ((line_end_x - line_start_x) * y2) + (line_end_x * line_start_y) - (
    #                     line_end_y * line_start_x)) / (
    #                            (line_end_y - line_start_y) ** 2 + (line_end_x - line_start_x) ** 2) ** 0.5
    #     if distance < radius2:
    #         return True
    #     else:
    #         return False

    # elif len(element) == 4:
    #     x1, y1, w1, h1 = element
    #     print(element)
    #     circle_x, circle_y, circle_radius = x2, y2, radius2
    #     rect_x1 = x1 - w1 / 2
    #     rect_y1 = y1 - h1 / 2
    #     rect_x2 = x1 + w1 / 2
    #     rect_y2 = y1 + h1 / 2
    #     print(rect_x1, rect_y1, rect_x2, rect_y2)
    #     # Find the distance between center of circle and closest point on rectangle
    #     dx = abs(circle_x - max(rect_x1, min(circle_x, rect_x2)))
    #     dy = abs(circle_y - max(rect_y1, min(circle_y, rect_y2)))
    #     print(math.sqrt(dx ** 2 + dy ** 2), circle_radius)
    #     if dx ** 2 + dy ** 2 <= circle_radius ** 2:
    #         return True
    #     else:
    #         return False

    # elif len(element) == 4:
    #     # Contributor: Kaiyuan Tan
    #     # box detection check
    #     x1, y1, w1, h1 = element 
    #     cuboid_length = h1
    #     cuboid_width = w1
    #     global_distance = 0
    #     cuboid_center = (x1, y1)
    #     ball_center = (x2, y2)

    #     # Closest point on the rec
    #     closest_point = [0, 0]
    #     # Check x coordinate
    #     closest_point[0] = max(cuboid_center[0] - cuboid_length / 2,
    #                            min(ball_center[0], cuboid_center[0] + cuboid_length / 2))
    #     # Check y coordinate
    #     closest_point[1] = max(cuboid_center[1] - cuboid_width / 2,
    #                            min(ball_center[1], cuboid_center[1] + cuboid_width / 2))

    #     global_distance = math.sqrt((closest_point[0] - ball_center[0]) ** 2 + (closest_point[1] - ball_center[1]) ** 2 )

    #     if global_distance <= radius2:
    #         return True 
    #     else:
    #         return False
    # else:    
    #     raise NotImplementedError


def all_collision_inspect_old_discarded(element, x2, y2, radius2=0.2012):
    # !THIS FUNCTION IS NOW NOT IN USE SINCE THE LINE SEGMENT PART IS INACCURATE!
    # 
    # The turtlebot has width of 0.302 (adding 2 wheels); length/height of 0.266
    # x2 = x2 - 0.064  # offset (-0.064) 
    if len(element) == 3:
        # cylinder (circle in 2d) inspection
        x1, y1, radius1 = element
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if distance < radius1 + radius2:
            return True
        else:
            return False
    elif len(element) == 4:
        # line segment inspection
        x1, y1, w1, h1 = element
        line_start_x, line_start_y, line_end_x, line_end_y = 0, 0, 0, 0
        if w1 >= h1:
            line_start_x, line_start_y = x1 - w1, y1
            line_end_x, line_end_y = x1 + w1, y1

        else:
            line_start_x, line_start_y = x1, y1 + h1
            line_end_x, line_end_y = x1, y1 - h1
        distance = abs(
            ((line_end_y - line_start_y) * x2) - ((line_end_x - line_start_x) * y2) + (line_end_x * line_start_y) - (
                        line_end_y * line_start_x)) / (
                               (line_end_y - line_start_y) ** 2 + (line_end_x - line_start_x) ** 2) ** 0.5
        if distance < radius2:
            return True
        else:
            return False
    else:
        raise NotImplementedError
