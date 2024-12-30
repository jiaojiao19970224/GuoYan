import asyncio
import json
import joblib
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import math
import random
import time
import requests
from requests.exceptions import RequestException


def lowpass_filter(data, cutoff=0.1, fs=10.0, order=2):
    if len(data) <= order * 3:
        print("数据长度太短，无法进行滤波，返回原始数据。")
        return data
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def prepare_data_for_prediction(df):
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['heading'] = pd.to_numeric(df['heading'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['lat'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['lon'], errors='coerce')
    df['type'] = pd.to_numeric(df['type'], errors='coerce')

    df['speed_filtered'] = lowpass_filter(df['speed'], cutoff=0.1, fs=10.0, order=2)
    df['heading_filtered'] = lowpass_filter(df['heading'], cutoff=0.1, fs=10.0, order=2)

    df['acceleration'] = df['speed_filtered'].diff() * 10
    df['heading_rate_of_change'] = df['heading_filtered'].diff() * 10

    df['acceleration'].fillna(0, inplace=True)
    df['heading_rate_of_change'].fillna(0, inplace=True)

    df['lateral_acceleration'] = df['speed_filtered'] * (df['heading_rate_of_change'] * np.pi / 180)
    return df


def predict_intent(df, xgb_model, scaler):
    features = ['speed_filtered', 'heading_filtered', 'latitude', 'longitude', 'acceleration', 'heading_rate_of_change',
                'lateral_acceleration']
    X = df[features]
    X_scaled = scaler.transform(X)
    y_pred_intent = xgb_model.predict(X_scaled)
    return y_pred_intent


def determine_target_lane(lane_index, lane_num):
    """
    确定目标车道。
    """
    potential_target_lanes = []
    if lane_index > 0:
        potential_target_lanes.append(lane_index - 1)  # 左侧车道
    if lane_index < lane_num - 1:
        potential_target_lanes.append(lane_index + 1)  # 右侧车道
    return potential_target_lanes


def calculate_lane_change_necessity(S_PV, v_SV, t_THW):
    if v_SV == 0:
        return float('inf')
    return S_PV - v_SV * t_THW


def decide_lane_change_intent(trend_speed_benefit, trend_space_benefit, f_necessity,
                              current_lane, potential_target_lanes,
                              speed_threshold=1.0, space_threshold=4.0, necessity_threshold=2.0):
    best_intent = 1  # 1代表保持车道
    best_benefit = -float('inf')

    for target_lane in potential_target_lanes:
        if trend_speed_benefit > speed_threshold and trend_space_benefit > space_threshold and f_necessity > necessity_threshold:
            benefit = trend_speed_benefit + trend_space_benefit + f_necessity
            if benefit > best_benefit:
                best_benefit = benefit
                if target_lane < current_lane:
                    best_intent = 0  # 左换道
                elif target_lane > current_lane:
                    best_intent = 2  # 右换道

    return best_intent


def trend_analysis(values, window_size):
    if len(values) >= window_size:
        return np.mean(values[-window_size:])
    else:
        return np.mean(values)


def random_lane_change(force_interval=5):
    global last_random_change_time
    current_time = time.time()

    if current_time - last_random_change_time > force_interval:
        last_random_change_time = current_time
        return random.choice([0, 2])  # 随机换道
    return None


last_random_change_time = 0  # 初始化变量


class IntentRecognitionModule:
    def __init__(self, xgb_model, scaler):
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.vehicle_history_data = {}

    def process_vehicle_data(self, vehicle_id, realtime_data):

        if 'data' not in realtime_data or 'targets' not in realtime_data['data'] or len(
                realtime_data['data']['targets']) == 0:
            return {
                "targets": [],
                "timestamp": realtime_data.get("timestamp"),
                "fetchedData": realtime_data
            }

        timestamp = realtime_data['data'].get('timestamp')
        df = pd.DataFrame(realtime_data['data']['targets'])

        df['type'] = pd.to_numeric(df['type'], errors='coerce')

        results = []

        if not df.empty and vehicle_id in df['vehicleId'].values:
            host_vehicle_heading = df.loc[df['vehicleId'] == vehicle_id, 'heading'].values[0]

            other_vehicles = df.to_dict('records')

            for _, row in df.iterrows():
                if row['type'] not in [2, 5]:
                    continue

                if pd.isna(row['vehicleId']):
                    row['vehicleId'] = row['uuid']

                if 'uuid' in row.index:
                    row.drop('uuid', inplace=True)

                if row['vehicleId'] == vehicle_id:
                    current_data = {
                        "lat": float(row['lat']),
                        "lon": float(row['lon']),
                        "speed": float(row['speed']),
                        "timestamp": timestamp,
                        "laneIndex": int(row['laneIndex']),
                        "laneNum": int(row['laneNum'])
                    }

                    if vehicle_id not in self.vehicle_history_data:
                        self.vehicle_history_data[vehicle_id] = []
                    self.vehicle_history_data[vehicle_id].append(current_data)

                    if len(self.vehicle_history_data[vehicle_id]) > 50:
                        self.vehicle_history_data[vehicle_id] = self.vehicle_history_data[vehicle_id][-50:]

                    current_lat = current_data['lat']
                    current_lon = current_data['lon']
                    current_speed = current_data['speed']
                    current_lane = current_data['laneIndex']
                    lane_num = current_data['laneNum']

                    potential_target_lanes = determine_target_lane(current_lane, lane_num)
                    v_desired = 50 / 3.6
                    t_THW = 1.5

                    leading_vehicle, leading_vehicle_distance = None, None

                    v_PV = v_desired - random.uniform(5, 20)
                    S_PV = random.uniform(20, 80)

                    f_necessity = calculate_lane_change_necessity(S_PV, v_PV - current_speed, t_THW)

                    speeds = [data_point['speed'] for data_point in self.vehicle_history_data[vehicle_id]]
                    trend_space_benefit = trend_analysis([S_PV], 5)
                    trend_speed_benefit = trend_analysis([v_desired - current_speed], 5)

                    print(
                        f"Trend Speed Benefit: {trend_speed_benefit}, Trend Space Benefit: {trend_space_benefit}, Necessity: {f_necessity}")

                    predicted_intent = decide_lane_change_intent(trend_speed_benefit, trend_space_benefit, f_necessity,
                                                                 current_lane, potential_target_lanes)

                    random_intent = random_lane_change(force_interval=5)
                    if random_intent is not None:
                        predicted_intent = random_intent

                    results.append({
                        "vehicleId": row['vehicleId'],
                        "vehicleIntention": predicted_intent,
                        "isHostVehicle": True
                    })

                else:
                    vehicle_data_df = pd.DataFrame([row])
                    vehicle_data_df = prepare_data_for_prediction(vehicle_data_df)
                    y_pred_intent = predict_intent(vehicle_data_df, self.xgb_model, self.scaler)
                    row['vehicleIntention'] = y_pred_intent[0]
                    results.append({
                        "vehicleId": row['vehicleId'],
                        "vehicleIntention": int(row['vehicleIntention']),
                        "isHostVehicle": False
                    })

        response = {
            "targets": results,
            "timestamp": timestamp,
            "fetchedData": realtime_data
        }
        return response

def main():

    xgb_model = joblib.load('model/intent_recognition_model_2.pkl')
    scaler = joblib.load('model/scaler2.pkl')


    intent_module = IntentRecognitionModule(xgb_model, scaler)

    vehicle_id = "12345"
    realtime_data = {
        "data": {
            "timestamp": 1633046400000,
            "targets": [
                {
                    "vehicleId": "12345",
                    "lat": 31.2304,
                    "lon": 121.4737,
                    "speed": 15.0,
                    "heading": 90.0,
                    "type": 2
                },
                {
                    "vehicleId": "67890",
                    "lat": 31.2310,
                    "lon": 121.4740,
                    "speed": 10.0,
                    "heading": 90.0,
                    "type": 2
                }
            ]
        }
    }

    realtime_data1 = """
    {
        "code": 0,
        "data": {
            "objNum": 4,
            "targets": [
                {
                    "crossId": "73",
                    "heading": 277.6761,
                    "height": 160,
                    "lat": 29.6098089,
                    "length": 459,
                    "lon": 106.2906199,
                    "roadSegId": null,
                    "speed": 0.099999994,
                    "srcType": 3,
                    "type": 2,
                    "uuid": "97184f98d3e2089f89e4eb94",
                    "vehicleId": null,
                    "width": 179,
                    "laneIndex": 1,
                    "laneNum": 5
                },
                {
                    "crossId": "73",
                    "heading": 7.9404,
                    "height": 179,
                    "lat": 29.6096309,
                    "length": 160,
                    "lon": 106.2906075,
                    "roadSegId": "S73down",
                    "speed": 0.76,
                    "srcType": 3,
                    "type": 1,
                    "uuid": "4956b61c3bf38f4889e4ff1c",
                    "vehicleId": null,
                    "width": 80,
                    "laneIndex": 1,
                    "laneNum": 1
                },
                {
                    "crossId": "73",
                    "heading": 184.9875,
                    "height": null,
                    "lat": 29.6094882,
                    "length": null,
                    "lon": 106.2905825,
                    "roadSegId": "S73down",
                    "speed": 0,
                    "srcType": 1,
                    "type": 2,
                    "uuid": "1a04ab92e5af62ae89e117e4",
                    "vehicleId": "R1000004",
                    "width": null,
                    "laneIndex": 4,
                    "laneNum": 4
                },
                {
                    "crossId": "73",
                    "heading": 17.3516,
                    "height": 160,
                    "lat": 29.6096294,
                    "length": 459,
                    "lon": 106.2906078,
                    "roadSegId": "S73down",
                    "speed": 1.2099999,
                    "srcType": 3,
                    "type": 2,
                    "uuid": "b557929ad3e5746089e4b098",
                    "vehicleId": null,
                    "width": 179,
                    "laneIndex": 1,
                    "laneNum": 1
                }
            ],
            "timestamp": 1728890341994
        },
        "message": "success"
    }
    """
    vehicle_id1 = 'R1000004'
    realtime_data1 = json.loads(realtime_data1)
    output = intent_module.process_vehicle_data(vehicle_id1, realtime_data1)

    print(json.dumps(output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
