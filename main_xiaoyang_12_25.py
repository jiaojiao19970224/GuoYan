from re import S
import datetime
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
import time

# 忽略所有警告
warnings.filterwarnings("ignore")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 创建以当前时间为名称的文件夹
folder_name = f'folder_{current_time}'
os.makedirs(folder_name, exist_ok=True)  # 如果文件夹已存在，忽略错误

# 定义文件名并存储在文件夹中
filename = os.path.join(folder_name, f'vehicle_data_{current_time}.json')
filename3_3 = os.path.join(folder_name, f'3_3_data_{current_time}.json')
filename3_4and4_3 = os.path.join(folder_name, f'3_4and4_3_data_{current_time}.json')
filename4_4 = os.path.join(folder_name, f'4_4_{current_time}.json')

# start_location = carla.Location(x=-1030, y=-840, z=1.0) #左侧车道
# start_location = carla.Location(x=-1023, y=-908, z=1.0)

import controller
from task4_4_xiaoyang_12_25 import global_path_planner, waypoint_list_2_target_path, Vehicle_control, extract_local_path, \
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
            "start": carla.Location(x=-1023.32, y=-908.26, z=2),
            "end": carla.Location(x=-1238.68, y=1404.57, z=2),
            "lane": 1,  # 左车道
            "color": carla.Color(255, 0, 0)
        },
        # "obs_vehicle1": {
        #     "start": carla.Location(x=243.62, y=16.36, z=5.11),
        #     "end": carla.Location(x=-382.40, y=12.30, z=2.30),
        #     "lane": 1,  # 中间车道
        #     "color": carla.Color(0, 0, 255)
        # },
        # "obs_vehicle2": {
        #     "start": carla.Location(x=261.23, y=12.38, z=5.34),
        #     "end": carla.Location(x=-382.40, y=12.30, z=2.30),
        #     "lane": 2,  # 右车道
        #     "color": carla.Color(0, 255, 0)
        # },
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

    # 主车 待输入
    # ego_vehicle = test.respone_ego(start_location,client)
    vehicle_id = 'QD1E003P'

    # 对 ego_vehicle 进行单独操作
    ego_vehicle = vehicles["ego_vehicle"]  # 提取 ego_vehicle 对象

    intentRecognitionModule = IntentRecognitionModule(
        xgb_model=joblib.load(os.path.join('model', 'intent_recognition_model_2.pkl')),
        scaler=joblib.load(os.path.join('model', 'scaler2.pkl')),
    )
    try:
        while True:
            # if traci.simulation.getMinExpectedNumber() > 0:

            data = test.data_record(client, ego_vehicle, traci)
            with open(filename, 'a') as json_file:
                json.dump(data, json_file, indent=4)
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

            with open(filename3_3, 'a') as json_file:
                json.dump(data, json_file, indent=4)

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
            with open(filename3_4and4_3, 'a') as json_file:
                json.dump(result, json_file, indent=4)

            world.debug.draw_string(ego_vehicle.get_location(),
                                    f"vehicle intention: {hostIntention}\n vehilce behavior: {hostBehavior}")

            # ---------------4_4------------------

            # 4.3 输入行为决策信息
            start_time = time.perf_counter()

            # 获取最新的行为数据（这里假设通过某种接口实时更新）
            behavior_data_json_import = result  # 实时获取行为数据（替换为实际数据源）
            print("行为数据:", behavior_data_json_import)

            # 遍历每辆车并更新决策
            for vehicle_name, vehicle_agent in vehicles.items():
                # 从行为数据中提取决策信息
                behavior, vertical_behavior, front_vehicle_speed = get_realtime_decision(behavior_data_json_import,
                                                                                         vehicle_name)
                print("当前横向决策", behavior, "当前纵向决策", vertical_behavior, "前车速度", front_vehicle_speed)
                print("当前车辆", vehicle_name)
                # 横向决策：如果需要变道，则调用变道逻辑
                if behavior in [0, 2]:  # 0=左变道, 2=右变道
                    vehicle_agent.plan_lane_change(behavior)

                # 更新车辆状态并进行控制（加入纵向速度规划）
                vehicle_agent.update()
                vehicle_agent.control_step(vertical_behavior, front_vehicle_speed)
                vehicle_agent.draw_trajectory()  # 绘制车辆的轨迹

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"Algorithm 4_4 execution time {execution_time:.3f} ms")

    except KeyboardInterrupt:
        print("Manual interrupt received. Stopping simulation.")
    finally:
        traci.close()
        # 销毁所有车辆
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()

        print("Vehicle destroyed.")


if __name__ == '__main__':
    main()
