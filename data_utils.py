def preprocess_data(data_recieved):
    x = []
    y = []

    for item in data_recieved:
        y.append(item['label'])
        ap_map = dict()
        for ap in item['ap_list']:
            ap_map[ap['ssid_bssid']] = ap['signal_strength']
        x.append(ap_map)

    return x, y


def preprocess_predict_data(data_recieved):
    ap_map = dict()
    for ap in data_recieved:
        ap_map[ap['ssid_bssid']] = ap['signal_strength']

    return ap_map
