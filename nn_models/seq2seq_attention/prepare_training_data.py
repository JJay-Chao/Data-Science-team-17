import csv
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-action', type=str)
args = parser.parse_args()


# read file path
train_2_filepath = '../data/train_2_complement.csv'
webpage_handcraft_filepath = '../data/feature_handcraft.csv'
webpage_trend_seasonal_residual_filepath = '../data/feature_trend_seasonal_residual.csv'
webpage_country_filepath = '../data/country_onehot.csv'
webpage_agent_filepath = '../data/agent_onehot.csv'
webpage_device_filepath = '../data/device_onehot.csv'

# write file path
# time series files
training_data_series_filepath = '../data/series_data/train_64_64_step_32_series_data.csv'
validation_data_series_filepath = '../data/series_data/valid_64_64_step_32_series_data.csv'
infer_data_series_filepath = '../data/series_data/infer_64_64_step_32_series_data.csv'
select_pages_filepath = '../data/series_data/pageNames.csv'
traffic_maxmin_filepath = '../data/series_data/traffic_maxmin.csv'

# trend seasonal residual files
training_data_trend_filepath = '../data/series_data/train_64_64_step_32_trend_data.csv'
validation_data_trend_filepath = '../data/series_data/valid_64_64_step_32_trend_data.csv'
infer_data_trend_filepath = '../data/series_data/infer_64_64_step_32_trend_data.csv'
trend_maxmin_filepath = '../data/series_data/trend_maxmin.csv'

training_data_seasonal_filepath = '../data/series_data/train_64_64_step_32_seasonal_data.csv'
validation_data_seasonal_filepath = '../data/series_data/valid_64_64_step_32_seasonal_data.csv'
infer_data_seasonal_filepath = '../data/series_data/infer_64_64_step_32_seasonal_data.csv'
seasonal_maxmin_filepath = '../data/series_data/seasonal_maxmin.csv'

training_data_residual_filepath = '../data/series_data/train_64_64_step_32_residual_data.csv'
validation_data_residual_filepath = '../data/series_data/valid_64_64_step_32_residual_data.csv'
infer_data_residual_filepath = '../data/series_data/infer_64_64_step_32_residual_data.csv'
residual_maxmin_filepath = '../data/series_data/residual_maxmin.csv'

# country, agent, device files
training_data_country_filepath = '../data/series_data/train_country_data.csv'
training_data_agent_filepath = '../data/series_data/train_agent_data.csv'
training_data_device_filepath = '../data/series_data/train_device_data.csv'



def read_time_series(filepath):
    time_series = {}
    select_pages = []
    remove_pages = []
    remove_page_ids = []
    pointer = 0
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter=',')
        column_name = next(tsv_reader, None)

        for line in tsv_reader:
            pageID = line[0]
            pageName = line[-1]
            page_series = line[1:-1]
            page_series = [int(float(traffic)) for traffic in page_series]
            select_or_not = True
            if page_series == [0]*len(page_series):
                select_or_not = False

            if select_or_not:
                time_series[pageName] = []
                time_series[pageName].append(page_series)
                select_pages.append(pageName)
            else:
                remove_pages.append(pageName)
                remove_page_ids.append(pointer)

            pointer += 1

    return select_pages, remove_pages, remove_page_ids, time_series


def read_trend_seasonal_residual(filepath, time_series, remove_pages):
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter=',')
        column_name = next(tsv_reader, None)

        for line in tsv_reader:
            pageName = line[0]
            trend = line[1].split(' ')
            trend = [float(feature) for feature in trend]
            seasonal = line[2].split(' ')
            seasonal = [float(feature) for feature in seasonal]
            residual = line[3].split(' ')
            residual = [float(feature) for feature in residual]

            select_or_not = True
            if pageName in remove_pages:
                select_or_not = False

            if select_or_not:
                time_series[pageName].append(trend)
                time_series[pageName].append(seasonal)
                time_series[pageName].append(residual)

    return time_series


def read_country_agent_device(filepath, remove_page_ids, select_pages):
    one_hot_feature = {}
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        pointer = 0
        for index, line in enumerate(tsv_reader):
            select_or_not = True
            if index in remove_page_ids:
                select_or_not = False

            if select_or_not:
                one_hot_list = line[0].split(',')
                one_hot_list = [str(int(float(value))) for value in one_hot_list]
                one_hot_string = ' '.join(one_hot_list)
                one_hot_feature[select_pages[pointer]] = one_hot_string
                pointer += 1

    return one_hot_feature


def normalization(time_series):
    traffic_maxmin = []
    trend_maxmin = []
    seasonal_maxmin = []
    residual_maxmin = []
    for pageName, values in time_series.items():
        # normalize
        traffic_series = values[0]
        traffic_max = max(traffic_series)
        traffic_min = min(traffic_series)
        traffic_maxmin.append([traffic_max, traffic_min])
        for id in range(len(traffic_series)):
            x_std = (traffic_series[id] - traffic_min) / (traffic_max - traffic_min)
            traffic_series[id] = (x_std * (1-(-1))) + (-1)

        time_series[pageName][0] = traffic_series

        # normalize
        trend_series = values[1]
        trend_max = max(trend_series)
        trend_min = min(trend_series)
        trend_maxmin.append([trend_max, trend_min])
        for id in range(len(trend_series)):
            x_std = (trend_series[id] - trend_min) / (trend_max - trend_min)
            trend_series[id] = (x_std * (1-(-1))) + (-1)

        time_series[pageName][1] = trend_series

        # normalize
        seasonal_series = values[2]
        seasonal_max = max(seasonal_series)
        seasonal_min = min(seasonal_series)
        seasonal_maxmin.append([seasonal_max, seasonal_min])
        for id in range(len(seasonal_series)):
            x_std = (seasonal_series[id] - seasonal_min) / (seasonal_max - seasonal_min)
            seasonal_series[id] = (x_std * (1-(-1))) + (-1)

        time_series[pageName][2] = seasonal_series

        # normalize
        residual_series = values[3]
        residual_max = max(residual_series)
        residual_min = min(residual_series)
        residual_maxmin.append([residual_max, residual_min])
        for id in range(len(residual_series)):
            x_std = (residual_series[id] - residual_min) / (residual_max - residual_min)
            residual_series[id] = (x_std * (1-(-1))) + (-1)

        time_series[pageName][3] = residual_series

    return time_series, traffic_maxmin, trend_maxmin, seasonal_maxmin, residual_maxmin


def distribute_series_data(time_series, interval=64):
    # time_series dictionary to array
    time_series_array = []
    trend_array = []
    seasonal_array = []
    residual_array = []
    for key, values in time_series.items():
        time_series_array.append(values[0])
        trend_array.append(values[1])
        seasonal_array.append(values[2])
        residual_array.append(values[3])

    time_series_array = np.array(time_series_array)
    trend_array = np.array(trend_array)
    seasonal_array = np.array(seasonal_array)
    residual_array = np.array(residual_array)

    # prepare training & development set
    # time series
    train_time_series_data = time_series_array[:, :time_series_array.shape[1]-128]
    valid_time_series_data = time_series_array[:, time_series_array.shape[1]-128:]

    # trend
    train_trend_data = trend_array[:, :trend_array.shape[1]-128]
    valid_trend_data = trend_array[:, trend_array.shape[1]-128:]

    # seasonal
    train_seasonal_data = seasonal_array[:, :seasonal_array.shape[1]-128]
    valid_seasonal_data = seasonal_array[:, seasonal_array.shape[1]-128:]

    # residual
    train_residual_data = residual_array[:, :residual_array.shape[1]-128]
    valid_residual_data = residual_array[:, residual_array.shape[1]-128:]

    # prepare infer data
    # time series
    infer_time_series_data = time_series_array.tolist()
    for id in range(len(infer_time_series_data)):
        infer_time_series_data[id] = infer_time_series_data[id][len(infer_time_series_data[id])-interval:]

    # trend
    infer_trend_data = trend_array.tolist()
    for id in range(len(infer_trend_data)):
        infer_trend_data[id] = infer_trend_data[id][len(infer_trend_data[id])-interval:]

    # seasonal
    infer_seasonal_data = seasonal_array.tolist()
    for id in range(len(infer_seasonal_data)):
        infer_seasonal_data[id] = infer_seasonal_data[id][len(infer_seasonal_data[id])-interval:]

    # residual
    infer_residual_data = residual_array.tolist()
    for id in range(len(infer_residual_data)):
        infer_residual_data[id] = infer_residual_data[id][len(infer_residual_data[id])-interval:]

    return (train_time_series_data, valid_time_series_data, infer_time_series_data), (train_trend_data, valid_trend_data, infer_trend_data), (train_seasonal_data, valid_seasonal_data, infer_seasonal_data), (train_residual_data, valid_residual_data, infer_residual_data)


def prepare_series_input_output_pair(time_series, select_pages, interval=64, step=32):
    # array to list
    time_series = time_series.tolist()

    data_pairs = {}
    for pageName, web_traffic in zip(select_pages, time_series):
        pointer=0
        data_pairs[pageName] = []
        while (pointer+interval+interval) <= len(web_traffic):
            inp = web_traffic[pointer:pointer+interval]
            tgt = web_traffic[pointer+interval:pointer+interval+interval]
            data_pairs[pageName].append((inp, tgt))
            pointer += step

    return data_pairs


def write_time_series_file(select_pages_path, train_path, valid_path, infer_path, select_pages, train_data_pairs, valid_data_pairs, infer_data):
    with open(select_pages_path, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for page_name in select_pages:
            tsv_writer.writerow([page_name])

    with open(train_path, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for name, pairs in train_data_pairs.items():
            for pair in pairs:
                inp_list = [str(value) for value in pair[0]]
                inp_string = ' '.join(inp_list)
                tgt_list = [str(value) for value in pair[1]]
                tgt_string = ' '.join(tgt_list)
                tsv_writer.writerow([name, inp_string, tgt_string])

    with open(valid_path, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for name, pairs in valid_data_pairs.items():
            for pair in pairs:
                inp_list = [str(value) for value in pair[0]]
                inp_string = ' '.join(inp_list)
                tgt_list = [str(value) for value in pair[1]]
                tgt_string = ' '.join(tgt_list)
                tsv_writer.writerow([name, inp_string, tgt_string])

    with open(infer_path, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for series in infer_data:
            series_list = [str(value) for value in series]
            series_string = ' '.join(series_list)
            tsv_writer.writerow([series_string])


def write_trend_seasonal_residual_file(train_path, valid_path, infer_path, train_data_pairs, valid_data_pairs, infer_data):
    with open(train_path, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for name, pairs in train_data_pairs.items():
            for pair in pairs:
                inp_list = [str(value) for value in pair[0]]
                inp_string = ' '.join(inp_list)
                tsv_writer.writerow([name, inp_string])

    with open(valid_path, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for name, pairs in valid_data_pairs.items():
            for pair in pairs:
                inp_list = [str(value) for value in pair[0]]
                inp_string = ' '.join(inp_list)
                tsv_writer.writerow([name, inp_string])

    with open(infer_path, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for series in infer_data:
            series_list = [str(value) for value in series]
            series_string = ' '.join(series_list)
            tsv_writer.writerow([series_string])


def write_maxmin_files(traffic_filepath, trend_filepath, seasonal_filepath, residual_filepath, traffic_maxmin, trend_maxmin, seasonal_maxmin, residual_maxmin, select_pages):
    with open(traffic_filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for pageName, values in zip(select_pages, traffic_maxmin):
            tsv_writer.writerow([pageName, values[0], values[1]])

    with open(trend_filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for pageName, values in zip(select_pages, trend_maxmin):
            tsv_writer.writerow([pageName, values[0], values[1]])

    with open(seasonal_filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for pageName, values in zip(select_pages, seasonal_maxmin):
            tsv_writer.writerow([pageName, values[0], values[1]])

    with open(residual_filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for pageName, values in zip(select_pages, residual_maxmin):
            tsv_writer.writerow([pageName, values[0], values[1]])


def write_country_agent_device_files(filepath, data):
    with open(filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for name, one_hot in data.items():
            tsv_writer.writerow([name, one_hot])






if __name__ == '__main__':
    print('reading data...')
    select_pages, remove_pages, remove_page_ids, time_series = read_time_series(train_2_filepath)
    time_series = read_trend_seasonal_residual(webpage_trend_seasonal_residual_filepath, time_series, remove_pages)
    country_data = read_country_agent_device(webpage_country_filepath, remove_page_ids, select_pages)
    agent_data = read_country_agent_device(webpage_agent_filepath, remove_page_ids, select_pages)
    device_data = read_country_agent_device(webpage_device_filepath, remove_page_ids, select_pages)

    # normalize data
    time_series, traffic_maxmin, trend_maxmin, seasonal_maxmin, residual_maxmin = normalization(time_series)

    # distribute data
    print('distribut into train-valid-test...')
    time_series_data, trend_data, seasonal_data, residual_data = distribute_series_data(time_series, interval=64)

    print('prepare data pairs...')
    # time series
    train_time_series_pairs = prepare_series_input_output_pair(time_series_data[0], select_pages, interval=64, step=32)
    valid_time_series_pairs = prepare_series_input_output_pair(time_series_data[1], select_pages, interval=64, step=32)

    # trend
    train_trend_pairs = prepare_series_input_output_pair(trend_data[0], select_pages, interval=64, step=32)
    valid_trend_pairs = prepare_series_input_output_pair(trend_data[1], select_pages, interval=64, step=32)

    # seasonal
    train_seasonal_pairs = prepare_series_input_output_pair(seasonal_data[0], select_pages, interval=64, step=32)
    valid_seasonal_pairs = prepare_series_input_output_pair(seasonal_data[1], select_pages, interval=64, step=32)

    # residual
    train_residual_pairs = prepare_series_input_output_pair(residual_data[0], select_pages, interval=64, step=32)
    valid_residual_pairs = prepare_series_input_output_pair(residual_data[1], select_pages, interval=64, step=32)

    print('store data preprocessing result...')
    # write time series files
    write_time_series_file(select_pages_filepath, training_data_series_filepath, validation_data_series_filepath, infer_data_series_filepath, select_pages, train_time_series_pairs, valid_time_series_pairs, time_series_data[2])

    # write trend seasonal residual files
    write_trend_seasonal_residual_file(training_data_trend_filepath, validation_data_trend_filepath, infer_data_trend_filepath, train_trend_pairs, valid_trend_pairs, trend_data[2])
    write_trend_seasonal_residual_file(training_data_seasonal_filepath, validation_data_seasonal_filepath, infer_data_seasonal_filepath, train_seasonal_pairs, valid_seasonal_pairs, seasonal_data[2])
    write_trend_seasonal_residual_file(training_data_residual_filepath, validation_data_residual_filepath, infer_data_residual_filepath, train_residual_pairs, valid_residual_pairs, residual_data[2])

    # write normalization maxmin files
    write_maxmin_files(traffic_maxmin_filepath, trend_maxmin_filepath, seasonal_maxmin_filepath, residual_maxmin_filepath, traffic_maxmin, trend_maxmin, seasonal_maxmin, residual_maxmin, select_pages)

    # write country, agent, device files
    write_country_agent_device_files(training_data_country_filepath, country_data)
    write_country_agent_device_files(training_data_agent_filepath, agent_data)
    write_country_agent_device_files(training_data_device_filepath, device_data)
