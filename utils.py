import json
import requests

# def get_lane(lon, lat):
#     data = {
#         "lonLats": [
#             [lon, lat]
#         ]
#     }
#
#     response = requests.post(settings.GPS2LANE_URL, data=json.dumps(data),
#                              headers={'Content-Type': 'application/json'})
#     if response.status_code == 200:
#         result_info = json.loads(response.text)
#         if result_info['code'] == 200 and result_info['data'] is not None:
#             gps_lane_list = result_info['data']
#
#     return gps_lane_list


def getVehiclesInfoFromCloudAPI(message, typeList):
    """
    解析云平台回参，存储于vehicle_info
    :param message: 已经是解析后的字典格式的云平台响应
    :return:
    """
    try:
        vehicle_info = {}
        #print(f"Original message: {message}")
        #print(f"Type list: {typeList}")

        targets = message.get('data', {}).get('targets', [])
        #print(f"Extracted targets: {targets}")

        if targets:
            for target in targets:
                if 'vehicleId' in target and target['vehicleId'] is not None and str(target['type']) in typeList:
                    vehicleID = target['vehicleId']
                    lon = target['lon']
                    lat = target['lat']
                    # laneIndex = get_lane(lon, lat)
                    vehicle_info[vehicleID] = {
                        "heading": target['heading'],
                        "vehicleId": vehicleID,
                        "lon": lon,
                        "lat": lat,
                        "speed": target['speed'],
                        "laneIndex": target['laneIndex'],
                        "uuid": target['uuid'],
                        "laneCount": target['laneNum']
                    }
                # 非网联车
                # if ('vehicleId' not in target or target['vehicleId'] is None) and str(target['type']) in typeList:
                else:
                    vehicleID = target['uuid']
                    lon = target['lon']
                    lat = target['lat']
                    # laneIndex = get_lane(lon, lat)
                    vehicle_info[vehicleID] = {
                        "heading": target['heading'],
                        "vehicleId": None,
                        "lon": lon,
                        "lat": lat,
                        "speed": target['speed'],
                        "laneIndex": target['laneIndex'],
                        "uuid": target['uuid'],
                        "laneCount": target['laneNum']
                    }
                    #print(f"Updated vehicle_info for ID {vehicleID}: {vehicle_info[vehicleID]}")  # 调试信息
      #  return vehicle_info

    except Exception as e:
        print(f"Error processing data: {e}")
        return {}
    return vehicle_info

def gethostVehicleIntention(targets):
    # 此函数现在直接接受targets列表，不再尝试从字典中获取
    for vehicle_info in targets:
        if vehicle_info.get('isHostVehicle'):
            return vehicle_info['vehicleId'], vehicle_info['vehicleIntention']
    return None, None

def getVehiclesWithIntentions(targets):
    vehicles_with_intentions = []
    for car in targets:
        vehicles_with_intentions.append({
            "vehicle_id": car.get("vehicleId"),
            "intention": car.get("vehicleIntention")
        })
    return vehicles_with_intentions
