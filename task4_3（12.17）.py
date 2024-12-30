import math

class Task4_3:
    def __init__(self, vehicle_info, car_info_and_intention, lane_count, min_ttc, coopSet, topological_vehicles):
        self._vehicle_info = vehicle_info
        self._car_info_and_intention = car_info_and_intention
        self._lane_count = lane_count
        self._min_ttc = min_ttc
        self._coopSet = coopSet
        self._topological_vehicles = topological_vehicles

    # Getters and Setters for all attributes
    def get_vehicle_info(self):
        return self._vehicle_info

    def set_vehicle_info(self, vehicle_info):
        self._vehicle_info = vehicle_info

    def get_car_info_and_intention(self):
        return self._car_info_and_intention

    def set_car_info_and_intention(self, car_info_and_intention):
        self._car_info_and_intention = car_info_and_intention

    def get_lane_count(self):
        return self._lane_count

    def set_lane_count(self, lane_count):
        self._lane_count = lane_count

    def get_min_ttc(self):
        return self._min_ttc

    def set_min_ttc(self, min_ttc):
        self._min_ttc = min_ttc

    def get_coopSet(self):
        return self._coopSet

    def set_coopSet(self, coopSet):
        self._coopSet = coopSet

    def get_topological_vehicles(self):
        return self._topological_vehicles

    def set_topological_vehicles(self, topological_vehicles):
        self._topological_vehicles = topological_vehicles

    # Public Methods
    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = 6371000 * c  # 地球半径为6371公里，乘以1000转换为米
        return distance

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lambda = math.radians(lon2 - lon1)
        y = math.sin(delta_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360

    def determine_relative_position(self, host_bearing, other_bearing):
        relative_bearing = (other_bearing - host_bearing + 360) % 360
        return relative_bearing

    def extract_host_vehicle_info(self):
        # 直接遍历意图信息列表
        for vehicle_info in self._car_info_and_intention:
            if vehicle_info.get('isHostVehicle'):
                return vehicle_info['vehicleId'], vehicle_info['vehicleIntention']
        return None, None

    def find_front_vehicle_on_current_lane(self, host_vehicle_id):
        host_vehicle = self._vehicle_info.get(host_vehicle_id)
        if not host_vehicle:
            return None  # 如果找不到主车信息，返回None

        host_lane_index = host_vehicle['laneIndex']
        host_bearing = host_vehicle['heading']
        front_vehicle = None
        min_front_distance = float('inf')

        for vehicle_id, info in self._vehicle_info.items():
            if vehicle_id == host_vehicle_id or info['laneIndex'] != host_lane_index:
                continue

            other_bearing = self.calculate_bearing(host_vehicle['lat'], host_vehicle['lon'], info['lat'], info['lon'])
            distance = self.haversine(host_vehicle['lon'], host_vehicle['lat'], info['lon'], info['lat'])
            relative_bearing = self.determine_relative_position(host_bearing, other_bearing)

            if (0 <= relative_bearing <= 90) or (270 <= relative_bearing < 360):
                if distance < min_front_distance:
                    front_vehicle = vehicle_id
                    min_front_distance = distance

        return front_vehicle

    def determine_target_lane(self, host_lane_index, host_intention):
        if host_intention == 1:  # 直行
            return host_lane_index
        elif host_intention == 0:  # 左转
            return host_lane_index - 1
        elif host_intention == 2:  # 右转
            return host_lane_index + 1

    def find_nearest_vehicles_on_target_lane(self, host_vehicle_id, host_intention):
        host_vehicle = self._vehicle_info.get(host_vehicle_id)
        if not host_vehicle:
            return None, None, None  # 如果没有主车信息，返回None

        host_lane_index = host_vehicle['laneIndex'][0]['laneIndex'] if isinstance(host_vehicle['laneIndex'], list) else \
        host_vehicle['laneIndex']
        target_lane_index = self.determine_target_lane(host_lane_index, host_intention)
        host_bearing = host_vehicle['heading']

        target_front_vehicle = None
        target_rear_vehicle = None
        min_front_distance = float('inf')
        min_rear_distance = float('inf')

        for vehicle_id, info in self._vehicle_info.items():
            if vehicle_id == host_vehicle_id or info['laneIndex'] != target_lane_index:
                continue

            other_bearing = self.calculate_bearing(host_vehicle['lat'], host_vehicle['lon'], info['lat'], info['lon'])
            distance = self.haversine(host_vehicle['lon'], host_vehicle['lat'], info['lon'], info['lat'])
            relative_bearing = self.determine_relative_position(host_bearing, other_bearing)

            if (0 <= relative_bearing <= 90) or (270 <= relative_bearing < 360):
                if distance < min_front_distance:
                    target_front_vehicle = vehicle_id
                    min_front_distance = distance
            elif 90 < relative_bearing < 270 and distance < min_rear_distance:
                target_rear_vehicle = vehicle_id
                min_rear_distance = distance

        return target_lane_index, target_front_vehicle, target_rear_vehicle

    def get_relevant_lanes(self, host_lane_index, host_intention):
        if isinstance(host_lane_index, list):
            host_lane_index = host_lane_index[0]['laneIndex']
        relevant_lanes = []
        if host_intention == 1:  # 直行
            relevant_lanes = [host_lane_index]
        elif host_intention == 0:  # 左转
            relevant_lanes = [host_lane_index - 1, host_lane_index - 2]
        elif host_intention == 2:  # 右转
            relevant_lanes = [host_lane_index + 1, host_lane_index + 2]

        relevant_lanes = [lane for lane in relevant_lanes if 0 <= lane < self._lane_count]
        return relevant_lanes

    def check_min_ttc(self, host_vehicle_id, front_vehicle_id):
        if front_vehicle_id is None:
            return True

        host_vehicle = self._vehicle_info[host_vehicle_id]
        front_vehicle = self._vehicle_info[front_vehicle_id]

        distance = self.haversine(
            host_vehicle['lon'], host_vehicle['lat'],
            front_vehicle['lon'], front_vehicle['lat']
        )

        relative_speed = abs(host_vehicle['speed'] - front_vehicle['speed'])

        if relative_speed == 0:
            ttc = float('inf')
        else:
            ttc = distance / relative_speed

        return ttc >= self._min_ttc

    def determine_priority(self, coopSet, host_vehicle_id):
        host_data = self._vehicle_info[host_vehicle_id]
        host_lat = host_data['lat']
        host_lon = host_data['lon']
        host_bearing = host_data['heading']

        vehicle_positions = []

        for vehicle_id in coopSet:
            if vehicle_id == host_vehicle_id:
                vehicle_positions.append((vehicle_id, 0))
            else:
                other_data = self._vehicle_info[vehicle_id]
                other_lat = other_data['lat']
                other_lon = other_data['lon']
                bearing_to_other = self.calculate_bearing(host_lat, host_lon, other_lat, other_lon)
                relative_bearing = (bearing_to_other - host_bearing + 360) % 360

                if (0 <= relative_bearing <= 90) or (270 <= relative_bearing < 360):
                    vehicle_positions.append((vehicle_id, relative_bearing))
                else:
                    vehicle_positions.append((vehicle_id, relative_bearing))

        vehicle_positions.sort(key=lambda x: x[1])

        vehicle_behavior_priority = {}
        for index, (vehicle_id, _) in enumerate(vehicle_positions):
            vehicle_behavior_priority[vehicle_id] = {
                "behavior": 0 if vehicle_id != host_vehicle_id else host_data.get('vehicleIntention', 0),
                "priority": index + 1}

        return vehicle_behavior_priority

    def right_of_way_front(self, vehicle_id):
        vehicle_v = self._vehicle_info[vehicle_id]['speed']
        rou_human = 1.0
        a_min_break = 3.0
        a_max_break = 8.0
        a_max_acceleration = 2.0

        F_self = vehicle_v * rou_human + vehicle_v ** 2 / (2 * a_min_break) - vehicle_v ** 2 / (2 * a_max_break)
        F_consult = (vehicle_v * rou_human + a_max_acceleration * rou_human ** 2 / 2 +
                     (vehicle_v + a_max_acceleration * rou_human) ** 2 / (2 * a_min_break) - vehicle_v ** 2 / (
                             2 * a_max_break) - F_self)
        right_of_way_front = F_self + F_consult
        return right_of_way_front

    def right_of_way_behind(self, vehicle_id):
        vehicle_v = self._vehicle_info[vehicle_id]['speed']
        a_max_break = 5.0

        min_break_time_t = vehicle_v / a_max_break
        F_behind = vehicle_v * min_break_time_t - 0.5 * a_max_break * min_break_time_t ** 2
        return F_behind

    def compare_gap_and_right_of_way(self, leading_vehicle, following_vehicle):
        if leading_vehicle is None or following_vehicle is None:
            return 1

        leading_vehicle_pos = (self._vehicle_info[leading_vehicle]['lon'], self._vehicle_info[leading_vehicle]['lat'])
        following_vehicle_pos = (self._vehicle_info[following_vehicle]['lon'], self._vehicle_info[following_vehicle]['lat'])
        actual_gap = self.haversine(following_vehicle_pos[1], following_vehicle_pos[0],
                                    leading_vehicle_pos[1], leading_vehicle_pos[0])

        leading_vehicle_right_of_way = self.right_of_way_behind(leading_vehicle)
        following_vehicle_right_of_way = self.right_of_way_front(following_vehicle)

        total_right_of_way = leading_vehicle_right_of_way + following_vehicle_right_of_way

        if actual_gap > total_right_of_way:
            return 1
        else:
            return 0

    def check_conflict_vehicles(self, relevant_lanes, target_lane, host_lane_index, host_vehicle_id, host_intention):
        opposite_intention = 2 if host_intention == 0 else 0
        host_lat = self._vehicle_info[host_vehicle_id]['lat']
        host_lon = self._vehicle_info[host_vehicle_id]['lon']
        host_bearing = self._vehicle_info[host_vehicle_id]['heading']  # 确保使用正确的属性

        for vehicle_data in self._car_info_and_intention:  # 直接迭代列表
            vehicle_id = vehicle_data['vehicleId']
            vehicle_intention = vehicle_data['vehicleIntention']

            if vehicle_id in self._topological_vehicles and vehicle_intention == opposite_intention:
                vehicle_info_data = self._vehicle_info.get(vehicle_id)
                if not vehicle_info_data:
                    continue  # 如果没有该车辆信息，继续下一循环

                vehicle_lane_index = vehicle_info_data['laneIndex']

                if vehicle_lane_index in relevant_lanes and vehicle_lane_index not in [target_lane, host_lane_index]:
                    vehicle_lat = vehicle_info_data['lat']
                    vehicle_lon = vehicle_info_data['lon']

                    other_bearing = self.calculate_bearing(host_lat, host_lon, vehicle_lat, vehicle_lon)
                    relative_bearing = self.determine_relative_position(host_bearing, other_bearing)

                    if (0 <= relative_bearing <= 90) or (270 <= relative_bearing < 360):
                        return 0
                    else:
                        return 1

        return 1  # 如果没有冲突的车辆

    def main(self):
        host_vehicle_id, host_intention = self.extract_host_vehicle_info()
        #print("hostID:", host_vehicle_id)
        #print("hostIntention:", host_intention)

        front_vehicle = self.find_front_vehicle_on_current_lane(host_vehicle_id)

        target_lane, target_front_vehicle, target_rear_vehicle = \
            self.find_nearest_vehicles_on_target_lane(host_vehicle_id, host_intention)

        host_lane_index = self._vehicle_info[host_vehicle_id]['laneIndex']
        #print("Host lanes:", host_lane_index)

        relevant_lanes = self.get_relevant_lanes(host_lane_index, host_intention)
        #print("Relevant lanes:", relevant_lanes)
        #print("Target lane:", target_lane)
        #print("Front vehicle on target lane:", target_front_vehicle)
        #print("Rear vehicle on target lane:", target_rear_vehicle)

        if front_vehicle:
            current_lane_ttc_satisfied = self.check_min_ttc(host_vehicle_id, front_vehicle)
        else:
            current_lane_ttc_satisfied = True

        if target_front_vehicle:
            target_lane_ttc_satisfied = self.check_min_ttc(host_vehicle_id, target_front_vehicle)
        else:
            target_lane_ttc_satisfied = True

        if target_rear_vehicle:
            target_lane_back_ttc_satisfied = self.check_min_ttc(target_rear_vehicle, host_vehicle_id)
        else:
            target_lane_back_ttc_satisfied = True

        if not current_lane_ttc_satisfied or not target_lane_ttc_satisfied or not target_lane_back_ttc_satisfied:
            #print("Either current lane or target lane does not satisfy the minimum TTC condition. Returning 0.")
            vehicle_behavior_priority = {host_vehicle_id: {"behavior": 1, "priority": 1}}
            return vehicle_behavior_priority
        else:
            if self.compare_gap_and_right_of_way(target_front_vehicle, target_rear_vehicle) == 1:
                #print("实际空隙更大")
                conflict_vehicle = self.check_conflict_vehicles(relevant_lanes, target_lane, host_lane_index, host_vehicle_id, host_intention)
                if conflict_vehicle == 1:
                    #print("没有冲突的车辆")
                    if self._coopSet:
                        vehicle_behavior_priority = self.determine_priority(self._coopSet, host_vehicle_id)
                        for vehicle_id in self._coopSet:
                            if vehicle_id == host_vehicle_id:
                                vehicle_behavior_priority[vehicle_id] = {"behavior": host_intention,
                                                                         "priority": vehicle_behavior_priority[vehicle_id][
                                                                             'priority']}
                            else:
                                vehicle_behavior_priority[vehicle_id] = {"behavior": 1,
                                                                         "priority": vehicle_behavior_priority[vehicle_id][
                                                                             'priority']}
                        return vehicle_behavior_priority
                    else:
                        #print("无响应协作方，执行单车变道")
                        vehicle_behavior_priority = {host_vehicle_id: {"behavior": host_intention, "priority": 1}}
                        return vehicle_behavior_priority
                else:
                    vehicle_behavior_priority = {host_vehicle_id: {"behavior": 1, "priority": 1}}
                    return vehicle_behavior_priority
            else:
                vehicle_behavior_priority = {host_vehicle_id: {"behavior": 1, "priority": 1}}
                return vehicle_behavior_priority


if __name__ == '__main__':
    pass
