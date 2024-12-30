from re import S
import time
from datetime import datetime
from multiprocessing import Process, Pipe
import task4_4
import utils
from task3_4 import Task3_4
from task4_3 import Task4_3
from task3_3 import IntentRecognitionModule
import json
import joblib
import os
import test
import carla
import traci
import warnings

# 忽略所有警告
current_name = int(time.time())
warnings.filterwarnings("ignore")
filename= f'vehicle_data_{current_name}.json'
filename3_3= f'3_3_data_{current_name}.json'
filename3_4and4_3= f'3_4and4_3_data_{current_name}.json'
filename4_4= f'4_4_{current_name}.json'

# start_location = carla.Location(x=-1030, y=-840, z=1.0) #左侧车道
# start_location = carla.Location(x=-1023, y=-908, z=1.0)

import controller
from task4_4 import global_path_planner, waypoint_list_2_target_path, Vehicle_control, extract_local_path, cartesian_to_frenet, lane_change_trajectory, frenet_to_cartesian, cartesian_to_path, draw_debug_path, speed_planning, find_nearest_path_point

def main():
    # 连接到CARLA服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    traci.init(9090)
    traci.setOrder(2) # number can be anything as long as each client gets its own number
    # while traci.simulation.getMinExpectedNumber() > 0:
    #     traci.simulationStep()
    # more traci commands

    # 获取世界和地图
    world = client.get_world()
    map = world.get_map()

    # 新 L 型地图，设置起点和终点
    start_location0 = carla.Location(x=-1019.90, y=-907.74, z=2)
    start_location1 = carla.Location(x=-1023.32, y=-908.26, z=2)
    end_location0 = carla.Location(x=-1235.31, y=1404.68, z=2)
    end_location1 = carla.Location(x=-1238.68, y=1404.57, z=2)

    # 获取起点和终点的最近路点
    start_waypoint0 = map.get_waypoint(start_location0, project_to_road=True)
    end_waypoint0 = map.get_waypoint(end_location0, project_to_road=True)

    start_waypoint1 = map.get_waypoint(start_location1, project_to_road=True)
    end_waypoint1 = map.get_waypoint(end_location1, project_to_road=True)

    # 获取路径规划器并计算路径
    global_route_plan = global_path_planner(world_map=map, sampling_resolution=2)  # 实例化全局规划器
    pathway0 = global_route_plan.search_path_way(origin=start_waypoint0.transform.location,
                                                 destination=end_waypoint0.transform.location)
    pathway1 = global_route_plan.search_path_way(origin=start_waypoint1.transform.location,
                                                 destination=end_waypoint1.transform.location)

    # 将路径点转换为 (x, y, theta, kappa) 格式
    global_frenet_path0 = waypoint_list_2_target_path(pathway0)
    global_frenet_path1 = waypoint_list_2_target_path(pathway1)

    print(len(global_frenet_path0))
    print(len(global_frenet_path1))

    print("路径0", global_frenet_path0)
    print("路径1", global_frenet_path1)

    # 创建并初始化车辆
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    spawn_point = start_waypoint0.transform
    spawn_point.location.z = 1.0
    ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if ego_vehicle is None:
        raise ValueError("Vehicle could not be spawned. Check spawn point and blueprint.")
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(spawn_point.location + carla.Location(z=50),
                                                                        carla.Rotation(pitch=-90)))

    # 控制器参数，max_speed 道路限速
    max_speed = 13.8   # m/s
    vehicle_para = (1.015, 2.910 - 1.015, 1412, -148970, -82204, 1537)
    controller = "MPC_controller"
    Controller = Vehicle_control(ego_vehicle=ego_vehicle, vehicle_para=vehicle_para,
                                 pathway=global_frenet_path0, controller_type=controller)

    # 主车 待输入
    # ego_vehicle = test.respone_ego(start_location,client)
    vehicle_id = 'QD1E003P'

    # 初始化路径状态
    current_path = global_frenet_path0
    global_index = 0
    is_in_lane_change = False
    transition_path = None
    target_path = None

    num = 0
    # 初始化行为变量，确保不会报错
    behavior = 1  # 假设行为初始化为 "0" 或者 "保持当前车道" 的行为

    global_index_lane = 0
    # 设置提取的路径长度 10 个点为 10 * 2 = 20 m
    change_lane_length = 10

    # 设置变道速度
    lane_change_speed = 11  # 变道时目标速度40 km/h

    intentRecognitionModule = IntentRecognitionModule(
            xgb_model=joblib.load(os.path.join('model', 'intent_recognition_model_2.pkl')),
            scaler=joblib.load(os.path.join('model', 'scaler2.pkl')),
        )
    try:
        while True:
            #if traci.simulation.getMinExpectedNumber() > 0:
            

            # 感知数据 待输入
            # data = '''
            # {
            #     "code": 0,
            #     "data": {
            #         "objNum": 1,
            #         "targets": [
            #             {
            #                 "crossId": "31",
            #                 "heading": 90.5,
            #                 "height": null,
            #                 "laneId": "2023041215214795678",
            #                 "laneNum": 4,
            #                 "lat": 29.5139267,
            #                 "length": null,
            #                 "lon": 106.3311464,
            #                 "roadSegId": "RS-1-S1right",
            #                 "speed": 0.0,
            #                 "srcType": 1,
            #                 "type": 2,
            #                 "uuid": "1062dbd7883581d4e17a2018",
            #                 "vehicleId": "QD1E003P",
            #                 "width": null,
            #                 "laneIndex": 3
            #             }
            #         ],
            #         "timestamp": 1730362993500
            #     },
            #     "message": "success"
            # }
            # '''
            data = test.data_record(client, ego_vehicle, traci)
            # with open(filename, 'a') as json_file:
            #     json.dump(data, json_file, indent=4)
            # ---------3_3------------------
            start_time = time.perf_counter()
            data = intentRecognitionModule.process_vehicle_data(vehicle_id, data)

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"Algorithm 3_3 execution time {execution_time:.3f} ms")

            
            if "fetchedData" not in data:
                print(json.dumps(data))
            message = data["fetchedData"]

            targets = data["targets"]
            host, hostIntention = utils.gethostVehicleIntention(targets)
            vehiclesWithIntentions = utils.getVehiclesWithIntentions(targets)

            typeList = {"2", "5"}
            vehicle_info = utils.getVehiclesInfoFromCloudAPI(message, typeList)

            # with open(filename3_3, 'a') as json_file:
            #     json.dump(data, json_file, indent=4)

            # ---------3_4------------------
            start_time = time.perf_counter()

            task = Task3_4(host)
            task.set_type_list(typeList)
            task.set_vehicle_info(vehicle_info)
            task.set_host_vehicle_intention(hostIntention)
            task.set_vehicles_with_intentions(vehiclesWithIntentions)

            coop_set = task.get_coope_set()

            topological_vehicles = task.get_topological_set()

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"Algorithm 3_4 execution time {execution_time:.3f} ms")

            # ---------------4_3------------------
            start_time = time.perf_counter()
            host_info = vehicle_info.get(host)
            lane_count = host_info["laneCount"]
            min_ttc = 3.0
            vehicle_behavior = Task4_3(vehicle_info, targets, lane_count, min_ttc, coop_set, topological_vehicles)
            vehicle_behavior_priority = vehicle_behavior.main()

            hostBehavior = vehicle_behavior_priority[vehicle_id]['behavior']

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"Algorithm 3_4 execution time {execution_time:.3f} ms")

            result = {
                'coop_set': list(coop_set),
                'topological_vehicles': [],
                'vehicle_behavior_priority': vehicle_behavior_priority,
            }

            for target in message['data']['targets']:
                if target["vehicleId"] in topological_vehicles:
                    result["topological_vehicles"].append(target)

            ###输出
            result = {
                'timestamp': data['timestamp'],
                'result': result
            }
            # with open(filename3_4and4_3, 'a') as json_file:
            #     json.dump(result, json_file, indent=4)

            world.debug.draw_string(ego_vehicle.get_location(), f"vehicle intention: {hostIntention}\n vehilce behavior: {hostBehavior}")

            # result['result']['vehicle_behavior_priority'][vehicle_id]['behavior'] = 1

            # ---------------4_4------------------
            json_data = '''
            {
                "code": 0,
                "data": {
                    "absFlag": null,
                    "accMode": null,
                    "accelPos": 6553.5,
                    "accelerationH": 0,
                    "accelerationV": 0,
                    "aebFlag": null,
                    "automatic": null,
                    "brakeFlag": 255,
                    "brakePos": 6553.5,
                    "categoryCode": null,
                    "daytimeRunning": null,
                    "dmsFlag": null,
                    "doors": null,
                    "driveMode": 255,
                    "espFlag": null,
                    "fcwFlag": null,
                    "fogLight": null,
                    "fuelConsumption": 655.35,
                    "hazardSignal": null,
                    "heading": 184.55,
                    "highBeam": null,
                    "lat": 29.602612,
                    "lcaFlag": null,
                    "ldwFlag": null,
                    "leftTurn": null,
                    "lkaFlag": null,
                    "lon": 106.2896636,
                    "lowBeam": null,
                    "mileage": null,
                    "parking": null,
                    "rightTurn": null,
                    "soc": null,
                    "speed": 3.78,
                    "tapPos": 0,
                    "tcsFlag": null,
                    "timestamp":  "1728890494475",
                    "vehFault": null,
                    "vehicleId": "R1000004"
                },
                "message": "success"
            }
            '''
            ego_data = json.loads(json_data)
            ego_data['data']['heading'] = host_info['heading']
            ego_data['data']['lat'] = host_info['lat']
            ego_data['data']['lon'] = host_info['lon']
            ego_data['data']['speed'] = host_info['speed']
            ego_data['data']['vehicleId'] = host_info['vehicleId']

            
            # 4.3 输入行为决策信息
            start_time_4_4 = time.perf_counter()
            behavior_data_json_import = result
            print("行为数据:", behavior_data_json_import)
            
            # 提取行为优先级信息
            behavior_priority = behavior_data_json_import.get("result", {}).get("vehicle_behavior_priority", {})
            print("提取的行为优先级:", behavior_priority)
            
            # 获取当前车辆的行为
            behavior = behavior_priority.get("QD1E003P", {}).get("behavior", None)
            vertical_behavior = behavior_priority.get("QD1E003P", {}).get("vertical_behavior", None)
            front_vehicle_speed = behavior_priority.get("QD1E003P", {}).get("front_vehicle_speed", None)
            print("当前横向行为:", behavior)
            # vertical_behavior 纵向决策 4 加速 5 保持当前速度 6 减速
            print("当前纵向决策:", vertical_behavior)
            # 前车速度，编导过程中前车不确定   # 输入的前车速度是 m/s 的单位
            if front_vehicle_speed is not None:
                print("当前前车速度:", front_vehicle_speed * 3.6)
            else:
                print("当前前车速度:", front_vehicle_speed)
            
            # 确认行为决策目标路径
            if behavior == 0:
                new_target_path = global_frenet_path0  # 切换到左车道
            elif behavior == 2:
                new_target_path = global_frenet_path1  # 切换到右车道
            else:
                new_target_path = current_path  # 保持当前车道

            if not is_in_lane_change and new_target_path != current_path:
                global_index_lane = global_index
                # print("变道前全局索引为：", global_index_lane)
                # 检查是否接近路径终点
                if global_index >= len(current_path) - 3:
                    print("接近路径终点，跳过变道规划")
                    continue

                print(
                    f"当前索引: {global_index}, 当前路径长度: {len(current_path)}, 目标路径长度: {len(new_target_path)}")

                # 1、提取局部路径，从当前路径的索引开始，取 20 个点，40m，格式为 x, y, theta, k
                # change_lane_length = 10  # 设置提取的路径长度
                local_path = extract_local_path(current_path, global_index, change_lane_length)

                # 2、执行变道逻辑，运用waypoint_list_2_target_path函数转换成 x, y, theta, k 形式
                # print(local_path)
                # local_frenet_path = waypoint_list_2_target_path(local_path)

                # 3、将笛卡尔路径转换为Frenet路径，artesian_to_frenet 函数，[(x1, y1, theta1, kappa1), (x2, y2, theta2, kappa2), ...] 转换成 [(s1, d1), (s2, d2) 格式
                frenetic_path = cartesian_to_frenet(local_path)
                print(frenetic_path)

                # 设置边界条件
                right = 3.5
                left = -3.5
                # choices = [right, left]

                s_ego = 0  # 起始弧长
                s_end = 20  # 终止弧长
                d_ego = 0  # 起始横向位移
                if behavior == 0:
                    d_end = left
                else:
                    # behavior == 2  右变道
                    d_end = right
                # d_end = random_choice = random.choice(choices)  # 终止横向位移，3米表示向左变道，-3米表示向右变道

                # 4、生成变道轨迹，lane_change_trajectory 函数，生成变道轨迹
                trajectory = lane_change_trajectory(frenetic_path, s_ego, s_end, d_ego, d_end)
                # print("变道轨迹", trajectory)

                # 执行路径转换
                cartesian_path = frenet_to_cartesian(local_path, trajectory)
                # print("s，d转换成x，y", cartesian_path)

                transition_path = cartesian_to_path(cartesian_path)
                # print("x，y转换成x, y, theta, k", transition_path)

                # 执行变道规划
                target_path = new_target_path
                # transition_path_xy, target_path = trajectory_planning(global_index, current_path, new_target_path)
                # transition_path = xy_list_2_target_path(transition_path_xy)

                if transition_path:
                    print("生成变道轨迹", transition_path)
                    current_path = transition_path
                    global_index = 0  # 重置索引到变道轨迹的起点
                    is_in_lane_change = True

                    # 实例化控制器，使其跟踪变道轨迹
                    Controller = Vehicle_control(
                        ego_vehicle=ego_vehicle,
                        vehicle_para=vehicle_para,
                        pathway=current_path,  # 设置为变道轨迹
                        controller_type=controller
                    )
                    draw_debug_path(world, transition_path, color=(255, 0, 0))  # 绘制变道轨迹

                # 获取当前速度
            current_speed = ego_vehicle.get_velocity().length()
            # print("当前速度为：", current_speed * 3.6)
            print("当前速度为：", current_speed)

            # 调用速度规划函数
            # new_speed, target_speed = speed_planning(behavior, current_speed, max_speed, is_in_lane_change, a_max=3, a_min=-4.5,
            #                            lane_change_speed=lane_change_speed)
            new_speed, target_speed = speed_planning(vertical_behavior, current_speed, max_speed, is_in_lane_change, a_max=3, a_min=-4.5,
                                                     lane_change_speed=lane_change_speed, front_vehicle_speed=front_vehicle_speed)
            print("目标的速度：", target_speed)
            # print("速度规划的速度：", new_speed * 3.6)
            print("速度规划的速度：", new_speed)

            # 控制车辆沿当前路径行驶
            control = Controller.run_step(target_speed=new_speed * 3.6)
            # control_details = {
            #             'steer': control.steer,
            #             'speed': new_speed,
            #             'throttle': control.throttle,
            #             'timestamp': result['timestamp']
            #
            #
            #     }
            # with open(filename4_4, 'a') as json_file:
            #     json.dump(control_details, json_file, indent=4)

            ego_vehicle.apply_control(control)

            # 更新索引并检测终点
            vehicle_loc = ego_vehicle.get_transform().location
            global_index = find_nearest_path_point(vehicle_loc.x, vehicle_loc.y, current_path)
            print("当前索引是", global_index)

            # 运行时间测试
            end_time4_4 = time.perf_counter()
            execution_time4_4 = (end_time4_4 - start_time_4_4) * 1000
            print(f"Algorithm 4_4 execution time {execution_time4_4:.3f} ms")

            # 检查是否完成变道
            if is_in_lane_change and global_index >= len(current_path) - 1:
                current_path = target_path
                print("变道完成，切换到目标路径")
                global_index = global_index_lane + change_lane_length
                # 将当前索引往后的路径点作为新的轨迹、拼接轨迹 current_path_combination
                current_path_combination = target_path[global_index + 1:]
                print("变道后新轨迹的起点为", current_path_combination[0])
                # global_index = 0  # 重置索引到目标路径的起点
                is_in_lane_change = False

                # 更新控制器轨迹
                Controller = Vehicle_control(
                    ego_vehicle=ego_vehicle,
                    vehicle_para=vehicle_para,
                    pathway=current_path_combination,  # 设置为目标路径
                    controller_type=controller
                )
                draw_debug_path(world, current_path_combination, color=(0, 255, 0))  # 绘制目标路径

    except KeyboardInterrupt:
        print("Manual interrupt received. Stopping simulation.")
    finally:
        traci.close()
        if ego_vehicle is not None:
            ego_vehicle.destroy()
            print("Vehicle destroyed.")

if __name__ == '__main__':
    main()
