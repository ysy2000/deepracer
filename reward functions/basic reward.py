import math

MAX_SPEED_THRESHOLD = 3.0
MAX_ABS_STEERING_THRESHOLD = 30
MAX_SIGHT = 1.0

def dist(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def rect(r, theta):
    """
    theta in degrees
    returns tuple; (float, float); (x,y)
    """

    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    return x, y

def polar(x, y):
    """
    returns r, theta(degrees)
    """

    r = (x ** 2 + y ** 2) ** .5
    theta = math.degrees(math.atan2(y,x))
    return r, theta


def angle_mod_360(angle):
    """
    Maps an angle to the interval -180, +180.
    Examples:
    angle_mod_360(362) == 2
    angle_mod_360(270) == -90
    :param angle: angle in degree
    :return: angle in degree. Between -180 and +180
    """

    n = math.floor(angle/360.0)

    angle_between_0_and_360 = angle - n*360.0

    if angle_between_0_and_360 <= 180.0:
        return angle_between_0_and_360
    else:
        return angle_between_0_and_360 - 360


def get_waypoints_ordered_in_driving_direction(params):
    waypoints = params['waypoints']
    
    # waypoints are always provided in counter clock wise order
    if params['is_reversed']: # driving clock wise.
        return list(reversed(params['waypoints']))
    else: # driving counter clock wise.
        return waypoints


def up_sample(waypoints, factor):
    """
    Adds extra waypoints in between provided waypoints
    :param waypoints:
    :param factor: integer. E.g. 3 means that the resulting list has 3 times as many points.
    :return:
    """
    p = waypoints
    n = len(p)

    return [[i / factor * p[(j+1) % n][0] + (1 - i / factor) * p[j][0],
             i / factor * p[(j+1) % n][1] + (1 - i / factor) * p[j][1]] for j in range(n) for i in range(factor)]


def get_target_point(params):    
    waypoints = up_sample(get_waypoints_ordered_in_driving_direction(params), 20)

    car = [params['x'], params['y']]

    distances = [dist(p, car) for p in waypoints]
    min_dist = min(distances)
    i_closest = distances.index(min_dist)

    n = len(waypoints)

    waypoints_starting_with_closest = [waypoints[(i+i_closest) % n] for i in range(n)]

    sight = MAX_SIGHT
    
    r = params['track_width'] * sight
    
    is_inside = [dist(p, car) < r for p in waypoints_starting_with_closest]
    i_first_outside = is_inside.index(False)

    if i_first_outside < 0:  # this can only happen if we choose r as big as the entire track
        return waypoints[i_closest]

    return waypoints_starting_with_closest[i_first_outside]


def get_target_steering_degree(params):
    tx, ty = get_target_point(params)
    car_x = params['x']
    car_y = params['y']
    dx = tx-car_x
    dy = ty-car_y
    heading = params['heading']

    _, target_angle = polar(dx, dy)
    steering_angle = target_angle - heading

    return angle_mod_360(steering_angle)

def score_steer_to_point_ahead(params):
    marker_2 = 0.2 * params['track_width']
    marker_3 = 0.3 * params['track_width']
    marker_4 = 0.4 * params['track_width']
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    all_wheels_one_track = params['all_wheels_on_track']
    best_steering_angle = get_target_steering_degree(params)
    steering_angle = params['steering_angle']
    
    error = (steering_angle - best_steering_angle) / 60.0  # 60 degree is already really bad

    score = max( 1.0 - abs(error) , 1e-3)

    # control speed by ideal angle
    if abs(best_steering_angle) < 10:
        SPEED_THRESHOLD = MAX_SPEED_THRESHOLD
    elif best_steering_angle < 20:
        SPEED_THRESHOLD = 0.6 * MAX_SPEED_THRESHOLD
    elif best_steering_angle < 30:
        SPEED_THRESHOLD = 0.3 * MAX_SPEED_THRESHOLD
    else:
        SPEED_THRESHOLD = 0.1 * MAX_SPEED_THRESHOLD

    if speed > SPEED_THRESHOLD:
    reward *= 0.5
    
    # set reward by car position based on center
    if distance_from_center <= marker_2:
        reward = 1
    elif distance_from_center <= marker_3:
        reward = 0.8
    elif distance_from_center <= marker_4:
        reward = 0.5
    else:
        reward = 0.2
        
    if not all_wheels_one_track: # Penalize if the car goes off track
        reward *= 1e-3
       
    progress = 0.01 * params['progress']
    reward += progress
        
    score *= reward
    
    return max(score, 0.01) # optimizer is rumored to struggle with negative numbers and numbers too close to zero 



def reward_function(params):
    return float(score_steer_to_point_ahead(params))
