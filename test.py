from threading import main_thread
import carla
import time
import csv
import json
import random
import carla.libcarla
import math
import os
import numpy as np
from pyproj import Proj, transform
from task4_4 import Vehicle_control
sampling_interval = 0.1  # Ĭ�ϲɼ�Ƶ��


def xy_coordinates(lon, lat):
    # 
    # ʹ�ö�ά�Ĳ���ת��ģ�ͼ���Ŀ�����ꡣ
    # ����:
    # - x1, y1: ԭʼ����
    # - x0, y0: ƽ�Ʋ��� ��X �� ��Y
    # - m: �߶Ȳ���
    # - alpha: ��ת�Ƕȣ���λΪ����
    # ����:
    # - x2, y2: ת���������
    # 
    
    x0=0.333448196513114
    y0=-1.37207946320158
    m=0.000505157577406327
    alpha=0.0127853881961249
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    x2 = x0 + (1 + m) * (cos_alpha * lon - sin_alpha * lat)
    y2 = y0 + (1 + m) * (sin_alpha * lon + cos_alpha * lat)
    
    return x2, y2


def transform_coordinates(x1, y1):
    # 
    # ʹ�ö�ά�Ĳ���ת��ģ�ͼ���Ŀ�����ꡣ
    # ����:
    # - x1, y1: ԭʼ����
    # - x0, y0: ƽ�Ʋ��� ��X �� ��Y
    # - m: �߶Ȳ���
    # - alpha: ��ת�Ƕȣ���λΪ����
    # ����:
    # - x2, y2: ת���������
    # 
    
    x0 = -0.315719364467558
    y0 = 1.37553560670668
    m = -0.000504902522066630
    alpha = -0.0127853881961249
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    x2 = x0 + (1 + m) * (cos_alpha * x1 - sin_alpha * y1)
    y2 = y0 + (1 + m) * (sin_alpha * x1 + cos_alpha * y1)
    
    return x2, y2




def cal_heading_kappa(frenet_path_xy_list: list):
    dx_ = []
    dy_ = []
    for i in range(len(frenet_path_xy_list) - 1):
        dx_.append(frenet_path_xy_list[i + 1][0] - frenet_path_xy_list[i][0])
        dy_.append(frenet_path_xy_list[i + 1][1] - frenet_path_xy_list[i][1])
    # ����theta,���߷����
    # ����n�����ֵõ���ֻ��n-1����ֽ��������Ҫ����β����ظ���Ԫ��������ÿ���ڵ��dx,dy
    dx_pre = [dx_[0]] + dx_  # ��ǰ��dx_�ĵ�һλ
    dx_aft = dx_ + [dx_[-1]]  # ���dx_�����һλ
    dx = (np.array(dx_pre) + np.array(dx_aft)) / 2

    dy_pre = [dy_[0]] + dy_
    dy_aft = dy_ + [dy_[-1]]
    dy = (np.array(dy_pre) + np.array(dy_aft)) / 2
    theta = np.arctan2(dy, dx)  # np.arctan2�Ὣ�Ƕ������ڣ�-pi, pi��֮��
    # ��������
    d_theta_ = np.diff(theta)  # ��ּ���
    d_theta_pre = np.insert(d_theta_, 0, d_theta_[0])
    d_theta_aft = np.insert(d_theta_, -1, d_theta_[-1])
    d_theta = np.sin((d_theta_pre + d_theta_aft) / 2)  # ��Ϊd_theta�Ǹ�С������sin(d_theta)����d_theta,�����ֵ��
    ds = np.sqrt(dx ** 2 + dy ** 2)
    k = d_theta / ds

    return list(theta), list(k)

def draw_debug_path(world, path, color=(0, 255, 0), z_offset=0.5):
    for point in path:
        world.debug.draw_point(
            carla.Location(x=point[0], y=point[1], z=z_offset),
            size=0.1,
            color=carla.Color(r=color[0], g=color[1], b=color[2]),
            life_time=1
        )


def xy_list_2_target_path(pathway_xy_list: list):
    
    theta_list, kappa_list = cal_heading_kappa(pathway_xy_list)
    target_path = []
    for i in range(len(pathway_xy_list)):
        target_path.append((pathway_xy_list[i][0], pathway_xy_list[i][1], theta_list[i], kappa_list[i]))
    return target_path


def convert_loc(x,y):

    # ����WGS84
    wgs84 = Proj(init='epsg:4326')  # WGS84 geographic coordinate system
    y =-y
    # ��������ī����ͶӰͶӰtemrc����UTM��ͶӰ��������ȡ��RoadRunner���ĵ㣬�������ģ�World Projections(Proj)
    tmerc = Proj(proj='tmerc',zone=48,ellps='WGS84',lat_0=29.60049390792847,lon_0=106.3007712364197,k=1,x_0=0,y_0=0,units='m')

    # ��WGS84ת����temrc
    lon,lat = transform(tmerc, wgs84, y, x)
 
    
    return lon,lat

def convert_latlon(lon,lat):

    # ����WGS84
    wgs84 = Proj(init='epsg:4326')  # WGS84 geographic coordinate system

    # ��������ī����ͶӰͶӰtemrc����UTM��ͶӰ��������ȡ��RoadRunner���ĵ㣬�������ģ�World Projections(Proj)
    tmerc = Proj(proj='tmerc',zone=48,ellps='WGS84',lat_0=29.60049390792847,lon_0=106.3007712364197,k=1,x_0=0,y_0=0,units='m')

    # ��WGS84ת����temrc
    x, y = transform(wgs84, tmerc, lon, lat)
    return x,-y
 
    # print(f"Temrc Coordinates��׼��: X={x}, Y={y}")
    # print(f"Temrc Coordinates����: X1={x1}, Y1={y1}")

    #�Ա�˼·
    #RoadRunner��γ��_�Ա�_shp��γ��
    #RoadRunnerƽ������_�Ա�_UEƽ�����꣨UE��λ���ף�Y��ȡ����


def lane_count(waypoint):
    lane_count = 1 
    right_lane = waypoint.get_right_lane()
    while right_lane and right_lane.lane_type == carla.LaneType.Driving:
        lane_count += 1
        right_lane = right_lane.get_right_lane()

    left_lane = waypoint.get_left_lane()
    while left_lane and left_lane.lane_type == carla.LaneType.Driving:
        lane_count += 1
        left_lane = left_lane.get_left_lane()
    return lane_count

def add_ackermann_control(received_data, vehicle):
    
    ackermann_control = carla.VehicleAckermannControl()
    send_steering = received_data["send_steering"]  # ת��ǣ���λΪ��
    send_acceleration = received_data["send_acceleration"]  # ���ٶȣ���λΪ m/s^2
    # ���ÿ��Ʋ���
    ackermann_control.steer = math.radians(send_steering)  # ת��Ϊ����
    ackermann_control.acceleration = send_acceleration  # ���ٶȣ�m/s^2��
    ackermann_control.speed = 20

    vehicle.apply_ackermann_control(ackermann_control)

def add_control(received_data, vehicle):
    
    vehicle_control = carla.VehicleControl()
    vehicle_control.throttle = received_data["send_Thtottel"]/75  # ����
    vehicle_control.brake = received_data["send_Brake"]/75  # ɲ��
    send_steer = received_data["send_steering"]
    # ���ÿ��Ʋ���
    vehicle_control.steer = send_steer/540  # ת��Ϊ-1 �� 1

    vehicle.apply_control(vehicle_control)



def respone_ego(start_location,client):
    world = client.get_world()
    map = world.get_map()
    Nearest_WayPoint=map.get_waypoint(start_location)           #�ҵ����ͣ����
    #���÷ų��ĸ߶�=4
    transform = Nearest_WayPoint.transform
    transform.location.z=1.0   #"�ų��ĸ߳�Ϊ1"
    #������ɵ�
    #transform = random.choice(world.get_map().get_spawn_points())  # �������һ������Ͷ�ŵ�
    #���ɳ���
    #blueprints_vehicle = world.get_blueprint_library().filter("vehicle.*")
    #vehicle=world.spawn_actor(random.choice(blueprints_vehicle), transform)
    #ָ������
    blueprints_vehicle = world.get_blueprint_library().find('vehicle.nissan.patrol_2021')
    #blueprints_vehicle = world.get_blueprint_library().find('vehicle.mini.cooper_s_2021')
    #blueprints_vehicle = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_vehicle=world.spawn_actor(blueprints_vehicle, transform)
    ##ego_vehicle.set_autopilot(enabled=True)
    #vehicle=world.spawn_actor(blueprints_vehicle, ego_spawn_point)
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                                        carla.Rotation(pitch=-90)))
    return ego_vehicle

def data_record(client, obs_actor1, ego_vehicle:list, traci):
    traci.simulationStep()
    world = client.get_world()
    map = world.get_map()
    current_time = time.time()
    # ��ȡ���г���
    world_actors = world.get_actors().filter('vehicle.*')
    num_vehicles = len(world_actors)
    vehicles_data = []
    ego_mark = None

    if num_vehicles > 0:
        print(f"{num_vehicles}cars detected, start recording")
        while True:
            # actors = world.get_actors()
            # vehs = actors.filter('vehicle.*')

            # for veh in vehs:
            #     print(f"ID:{veh.id}, sumo_id:{veh.attributes.get('role_name')}")
            for vehicle in world_actors:
                # ��ȡ�����������Ϣ
                vehicle_id = vehicle.id
                print("vehicle ID", vehicle_id)
                if vehicle_id == ego_vehicle[0].id:
                    ego_mark = 'ego_vehicle'
                    # ego_mark = 'obs_vehicle1'
                    # spectator = world.get_spectator()
                    # transform = ego_vehicle.get_transform()
                    # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                    #                                                     carla.Rotation(pitch=-90)))md
                if vehicle_id == ego_vehicle[1].id:
                    ego_mark = 'obs_vehicle1'
                if vehicle_id == ego_vehicle[2].id:
                    ego_mark = 'obs_vehicle2'
                if vehicle_id == obs_actor1.id:
                    ego_mark = 'obs'
                #     continue
                transform = vehicle.get_transform()
                location = transform.location
                sumo_id = vehicle.attributes.get('role_name')
                waypoint = map.get_waypoint(location)
                x_pos, y_pos = location.x, location.y
                if(carla.VehicleWheelLocation.FL_Wheel):
                    wheel_steer_angle = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel )
                else:
                    wheel_steer_angle =0
                roadID = waypoint.road_id
                laneID = waypoint.lane_id
                #laneNum = lane_count(waypoint)

                if ego_mark==None:
                     speed = traci.vehicle.getSpeed(sumo_id)
                    # pass
                else:
                    velocity = vehicle.get_velocity()
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                #acceleration = vehicle.get_acceleration()
                #acc = math.sqrt(acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2)
                                 #����json
                geoLocation = map.transform_to_geolocation(location)
                lat1,lon1 = geoLocation.latitude,geoLocation.longitude
                #lon1,lat1 = convert_loc(location.x,location.y)   #geoLocation = map.transform_to_geolocation(location)
                lon,lat = transform_coordinates(lon1,lat1)
                # lat = 29.60807157684982
                # lon =106.29013767875198
                vehicle_data = {      
                            "heading": transform.rotation.yaw + 50,  # �����
                            "laneNum": 2,  # ���ó�������+1��Ϊ���������ο�����ͼ���պϣ�lanecount�����ò��ˣ�
                            "lat": lat,  # �滻Ϊʵ�ʵľ�γ������
                            "lon": lon,  # �滻Ϊʵ�ʵľ�γ������
                            "speed": speed,  # �ٶ�
                            "type": 2,  # Ĭ�ϳ�������
                            "uuid": vehicle_id,  # ����Ψһ�� UUID
                            "vehicleId": ego_mark,  # �����س���Ĭ��Ϊ None
                            "laneIndex": math.fabs(laneID),  # ���ڳ���
                            "sumo_id": sumo_id
            }
                if(location.distance(ego_vehicle[0].get_transform().location) <= 100):
                    
                    vehicles_data.append(vehicle_data)

                ego_mark = None

                        # �������ݵ�CSV�ļ�
                # with open('vehicle_data.csv', 'a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow([current_time, num_vehicles, vehicle_id, x_pos, y_pos, speed, acc,
                #                      wheel_steer_angle, roadID, laneID])

            json_data = {
                        "code": 0,
                        "data": {
                        "objNum": len(vehicles_data),  # targets ������
                        "targets": vehicles_data,  # �����б�
                        "timestamp": int(current_time * 1000)  # ���뼶ʱ���
                       },
                        "message": "success"
                    }
            # with open(f'vehicle_data_{int(current_time)}.json', 'a') as json_file:
            #     json.dump(json_data, json_file, indent=4)
            return json_data  # ÿsampling_interval����һ��



# wgs84 = Proj(init='epsg:4326')  # WGS84 geographic coordinate system
# lat = 29.6079832257
# lon = 106.2904075443

#     # ��������ī����ͶӰͶӰtemrc����UTM��ͶӰ��������ȡ��RoadRunner���ĵ㣬�������ģ�World Projections(Proj)
# tmerc = Proj(proj='tmerc',zone=48,ellps='WGS84',lat_0=29.60049390792847,lon_0=106.3007712364197,k=1,x_0=0,y_0=0,units='m')

#     # ��WGS84ת����temrc
# lat,lon = transform(tmerc, wgs84,x, y)
# print(f"Temrc Coordinates X={lat}, Y={lon}")
   


# #testing
# location = carla.Location(x=-345, y=-1560,z=1.0)
# client = carla.Client('localhost', 2000)
# client.set_timeout(10.0)
# ego = respone_ego(location,client)
# ego.set_simulate_physics(True)
# try:
#     while True:
#         received_data = {
#             "send_steering": 0,  # ת��ǣ���λΪ��
#             "send_acceleration": 10
#          }
#         add_control(received_data,ego)
#         velocity = ego.get_velocity()
#         speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
#         print(f"Vehicle Speed: {speed:.2f} m/s")

# except:
#     ego.destroy()



# def main():

#     try:
#         # ���ӵ�CARLA������
#         client = carla.Client('localhost', 2000)
#         client.set_timeout(10.0)
#         #client.load_world('Carla/Maps/town')

#         # ��ȡCARLA������
#         world = client.get_world()
#         spectator = world.get_spectator()
#         map = world.get_map()
#         print(map.name)
#         weather = world.get_weather()


#         #ָ�����ɵ�
#         #�ڸ߾���ͼ����ȷ����������λ�á�http://opendrive.bimant.com/
#         start_location = carla.Location(x=-582, y=1227,z=4.0)       #Y������Ҫȡ���� �߾���ͼ�ϵĵ���carla�����е����꣬Yֵ�Ƿ����
#         Nearest_WayPoint=map.get_waypoint(start_location)           #�ҵ����ͣ����


#         #���÷ų��ĸ߶�=4
#         transform=Nearest_WayPoint.transform
#         transform.location.z=1.0   #"�ų��ĸ߳�Ϊ1"


#         #������ɵ�
#         #transform = random.choice(world.get_map().get_spawn_points())  # �������һ������Ͷ�ŵ�

#         #���ɳ���
#         #blueprints_vehicle = world.get_blueprint_library().filter("vehicle.*")
#         #vehicle=world.spawn_actor(random.choice(blueprints_vehicle), transform)

#         #ָ������
#         blueprints_vehicle = world.get_blueprint_library().find('vehicle.nissan.patrol_2021')
#         #blueprints_vehicle = world.get_blueprint_library().find('vehicle.mini.cooper_s_2021')
#         #blueprints_vehicle = world.get_blueprint_library().find('vehicle.tesla.model3')

#         ego_vehicle=world.spawn_actor(blueprints_vehicle, transform)
#         #vehicle=world.spawn_actor(blueprints_vehicle, ego_spawn_point)


#         #vehicle.set_autopilot()
#         ego_vehicle.set_autopilot(enabled=True)
#         return ego_vehicle.id

#         #spectator.set_transform(transform)    #�ӽ��л��� ���ɳ�����λ��
#         while True:

#             current_time = time.time()
#             # ��ȡ���г���
#             world_actors = world.get_actors().filter('vehicle.*')
#             num_vehicles = len(world_actors)
#             vehicles_data = []
#             ego_mark = None

#             if num_vehicles > 0:
#                 print(f"{num_vehicles}cars detected, start recording")
#                 while True:
#                     for vehicle in world_actors:
#                         # ��ȡ�����������Ϣ
#                         vehicle_id = vehicle.id
#                         if vehicle_id == ego_vehicle.id:
#                             ego_mark = 'QD1E003P'
#                         transform = vehicle.get_transform()
#                         location = transform.location
#                         x_pos, y_pos = location.x, location.y
#                         wheel_steer_angle = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel )
#                         roadID = map.get_waypoint(transform.location).road_id
#                         laneID = map.get_waypoint(transform.location).lane_id
#                         velocity = vehicle.get_velocity()
#                         speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
#                         acceleration = vehicle.get_acceleration()
#                         acc = (acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2) ** 0.5

#                         #����json
                    
#                         geoLocation = map.transform_to_geolocation(location)
#                         vehicle_data = {

                            
#                             "heading": transform.rotation.yaw,  # �����
#                              "laneNum": laneID,  # ���ó��������Ϊ���������ο�������ݾ����ͼ�����߼���   ��û��ó���
#                             "lat": geoLocation.latitude,  # �滻Ϊʵ�ʵľ�γ������
#                             "lon": geoLocation.longitude,  # �滻Ϊʵ�ʵľ�γ������
#                             "speed": (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5,  # �ٶ�
#                             "type": 2,  # Ĭ�ϳ�������
#                             "uuid": vehicle_id,  # ����Ψһ�� UUID
#                             "vehicleId": ego_mark,  # �����س���Ĭ��Ϊ None
#                             "laneIndex": laneID  # ���ڳ���
#             }
#                         vehicles_data.append(vehicle_data)
#                         ego_mark = None

#                         # �������ݵ�CSV�ļ�
#                         with open('vehicle_data.csv', 'a', newline='') as file:
#                             writer = csv.writer(file)
#                             writer.writerow([current_time, num_vehicles, vehicle_id, x_pos, y_pos, speed, acc,
#                                              wheel_steer_angle, roadID, laneID])
#                     json_data = {
#                         "code": 0,
#                         "data": {
#                         "objNum": len(vehicles_data),  # targets ������
#                         "targets": vehicles_data,  # �����б�
#                         "timestamp": int(current_time * 1000)  # ���뼶ʱ���
#                        },
#                         "message": "success"
#                     }
#                     vehicles_data = []
#                     time.sleep(sampling_interval)  # ÿsampling_interval����һ��

#         while True:
#             spectator = world.get_spectator()
#             transform = ego_vehicle.get_transform()
#             spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
#                                                                         carla.Rotation(pitch=-90)))
#     finally:
#         #vehicle.destroy()
#         print("actor destroied")