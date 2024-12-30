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
filename = f'vehicle_data_{current_name}.json'
filename3_3 = f'3_3_data_{current_name}.json'
filename3_4and4_3 = f'3_4and4_3_data_{current_name}.json'
filename4_4 = f'4_4_{current_name}.json'

# start_location = carla.Location(x=-1030, y=-840, z=1.0) #左侧车道
# start_location = carla.Location(x=-1023, y=-908, z=1.0)

import controller
from task4_4 import global_path_planner, waypoint_list_2_target_path, Vehicle_control, extract_local_path, \
    cartesian_to_frenet, lane_change_trajectory, frenet_to_cartesian, cartesian_to_path, draw_debug_path, \
    speed_planning, find_nearest_path_point, VehicleAgent, get_realtime_decision


def main():
    # 连接到CARLA服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    traci.init(9090)
    traci.setOrder(2)  # number can be anything as long as each client gets its own number
    # while traci.simulation.getMinExpectedNumber() > 0:
    #     traci.simulationStep()
    # more traci commands

    # 获取世界和地图
    world = client.get_world()
    map = world.get_map()

    trajectory_points = []

    # 定义车辆参数
    vehicle_para = (1.015, 2.910 - 1.015, 1412, -148970, -82204, 1537)

    # 配置车辆的起点、终点和初始车道
    vehicles_config = {
        "ego_vehicle": {
            # "start": carla.Location(x=252.50, y=19.59, z=5),
            # "end": carla.Location(x=-382.40, y=12.30, z=2.30),
            "start": carla.Location(x=-1019.90, y=-907.74, z=2),
            "end": carla.Location(x=-1235.31, y=1404.68, z=2),
            "lane": 0,  # 左车道
            "color": carla.Color(255, 0, 0)
        },
        "obs_vehicle1": {
            "start": carla.Location(x=-1022.11, y=-893.03, z=2),
            "end": carla.Location(x=-1235.31, y=1404.68, z=2),
            "lane": 1,  # 中间车道
            "color": carla.Color(0, 0, 255)
        },
        "obs_vehicle2": {
            "start": carla.Location(x=-1025.63, y=-895.36, z=2),
            "end": carla.Location(x=-1238.68, y=1404.57, z=2),
            "lane": 2,  # 右车道
            "color": carla.Color(0, 255, 0)
        },
    }

    # 创建车辆
    blueprint_library = world.get_blueprint_library()
    vehicles = {}
    for name, config in vehicles_config.items():
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        transform = map.get_waypoint(config["start"]).transform
        # transform.location.z = 5.0  # 将 z 坐标调整到较低高度
        transform.location.z = 1.0  # 将 z 坐标调整到较低高度
        # print(config["start"])
        # print(transform)
        vehicle = world.spawn_actor(vehicle_bp, transform)
        vehicles[name] = VehicleAgent(
            name=name,
            vehicle=vehicle,
            world=world,
            map=map,
            start_location=config["start"],
            end_location=config["end"],
            vehicle_para=vehicle_para,
            initial_lane=config["lane"],  # 传入初始车道
            color=config["color"]
        )

    # 静止车辆1
    obs_vehicle_bp1 = world.get_blueprint_library().find('vehicle.tesla.model3')
    print(f"静置车辆类型{type(obs_vehicle_bp1)}")
    obs_vehicle_bp1.set_attribute('color', '0,0,255')
    obs_spawn_point1 = carla.Transform()
    obs_spawn_point1.location = carla.Location(x=-1062.35, y=-645.03, z=2)
    obs_spawn_point1.rotation = carla.Rotation(yaw=90)
    obs_actor1 = world.spawn_actor(obs_vehicle_bp1, obs_spawn_point1)  # type: carla.Vehicle
    #print(obs_actor1.id)

    # 主车 待输入
    # ego_vehicle = test.respone_ego(start_location,client)
    vehicle_id = 'obs_vehicle2'

    # 对 ego_vehicle 进行单独操作
    ego_vehicle = vehicles["obs_vehicle2"]  # 提取 ego_vehicle 对象

    intentRecognitionModule = IntentRecognitionModule(
        xgb_model=joblib.load(os.path.join('model', 'intent_recognition_model_2.pkl')),
        scaler=joblib.load(os.path.join('model', 'scaler2.pkl')),
    )
    try:
        while True:
            # if traci.simulation.getMinExpectedNumber() > 0:
            start_time = time.perf_counter()
            data = test.data_record(client, obs_actor1, [vehicles['ego_vehicle'], vehicles['obs_vehicle1'], vehicles['obs_vehicle2']], traci)
            # data = test.data_record(client, [vehicles['ego_vehicle'], vehicles['obs_vehicle1'], vehicles['obs_vehicle2']], traci)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            #print(f"Algorithm RSU execution time {execution_time:.3f} ms")
            # with open(filename, 'a') as json_file:
            #     json.dump(data, json_file, indent=4)
            # ---------3_3------------------
            start_time = time.perf_counter()
            data = intentRecognitionModule.process_vehicle_data(vehicle_id, data)

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            #print(f"Algorithm 3_3 execution time {execution_time:.3f} ms")

            if "fetchedData" not in data:
                print(json.dumps(data))
            message = data["fetchedData"]

            targets = data["targets"]
            host, hostIntention = utils.gethostVehicleIntention(targets)
            vehiclesWithIntentions = utils.getVehiclesWithIntentions(targets)

            typeList = {"2", "5"}
            vehicle_info = utils.getVehiclesInfoFromCloudAPI(message, typeList)
            print("车辆数据")
            print(vehicle_info)

            # with open(filename3_3, 'a') as json_file:
            #     json.dump(data, json_file, indent=4)

            # ---------3_4------------------
            start_time = time.perf_counter()

            task = Task3_4(host)
            task.set_type_list(typeList)
            task.set_vehicle_info(vehicle_info)
            task.set_host_vehicle_intention(hostIntention)
            task.set_vehicles_with_intentions(vehiclesWithIntentions)

            # coop_set = task.get_coope_set()
            #
            # topological_vehicles = task.get_topological_set()
            coop_set = task.get_coopeset_1225()
            actors = world.get_actors()
            for v_id in coop_set:
                for actor in actors:
                    print(f"actorID：{actor.id} 协作车id：{v_id}")
                    # print(f"(atorlass({type(actor)}")
                    if actor.id == vehicle_info.get(v_id)['uuid']:
                        world.debug.draw_string(actor.get_transform().location, "coop", color=carla.Color(r=0, g=0, b=255))

            topological_vehicles = task.get_keyVehicles_set()
            print(f"关键车：{topological_vehicles} 意图：{hostIntention}")
            for v_id in topological_vehicles:
                for actor in actors:
                    print(f"actorID：{actor.id} 关键id：{v_id}")
                    if actor.id == vehicle_info.get(v_id)['uuid']:
                        world.debug.draw_string(actor.get_transform().location, "keyVehicle", color=carla.Color(r=0, g=255, b=0))

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            #print(f"Algorithm 3_4 execution time {execution_time:.3f} ms")

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
            #print(f"Algorithm 3_4 execution time {execution_time:.3f} ms")

            result = {
                'coop_set': list(coop_set),
                'topological_vehicles': [],
                'vehicle_behavior_priority': vehicle_behavior_priority,
            }

            for target in message['data']['targets']:
                if target["vehicleId"] in topological_vehicles:
                    result["topological_vehicles"].append(target)

            # 输出
            result = {
                'timestamp': data['timestamp'],
                'result': result
            }
            # with open(filename3_4and4_3, 'a') as json_file:
            #     json.dump(result, json_file, indent=4)
            print(f"(zlass({type(ego_vehicle)}")
            world.debug.draw_string(ego_vehicle.get_location(),
                                    f"vehicle intention: {hostIntention}\n vehilce behavior: {hostBehavior}")

            # ---------------4_4------------------

            # 4.3 输入行为决策信息
            start_time = time.perf_counter()

            # 获取最新的行为数据（这里假设通过某种接口实时更新）
            behavior_data_json_import = result  # 实时获取行为数据（替换为实际数据源）
            #print("行为数据:", behavior_data_json_import)

            # # 遍历每辆车并更新决策
            for vehicle_name, vehicle_agent in vehicles.items():
                # 从行为数据中提取决策信息
                behavior, vertical_behavior, front_vehicle_speed = get_realtime_decision(behavior_data_json_import,
                                                                                         str(vehicle_name))
                #print("当前车辆", vehicle_name, "当前横向决策", behavior, "当前纵向决策", vertical_behavior, "前车速度", front_vehicle_speed)
                # print("当前车辆", vehicle_name)
                # 横向决策：如果需要变道，则调用变道逻辑
                if behavior in [0, 2]:  # 0=左变道, 2=右变道
                    vehicle_agent.plan_lane_change(behavior)

                # 更新车辆状态并进行控制（加入纵向速度规划）
                vehicle_agent.update()
                # vehicle_agent.control_step(vertical_behavior, front_vehicle_speed)
                # vertical_behavior = 1
                # front_vehicle_speed = 5
                vehicle_agent.control_step(vertical_behavior, front_vehicle_speed)
                vehicle_agent.draw_trajectory()  # 绘制车辆的轨迹

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            #print(f"Algorithm 4_4 execution time {execution_time:.3f} ms")

    except KeyboardInterrupt:
        print("Manual interrupt received. Stopping simulation.")
        # 销毁所有车辆
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()
    finally:
        traci.close()
        # 销毁所有车辆
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()

        #print("Vehicle destroyed.")


if __name__ == '__main__':
    main()
