from datetime import datetime
import re
import os


def find_ohlcv_groups(features):
    # Initialize an empty dictionary to store the groups
    pattern = re.compile(r"^(.*?)(open|high|low|close|volume)(.*)$")

    # Initialize a dictionary to hold the matched groups
    groups = {}
    matched_features = set()

    # Iterate over the features to match and group them
    for feature in features:
        match = pattern.match(feature)
        if match:
            # Extract the base (prefix and suffix), keyword, and suffix
            prefix, keyword, suffix = match.groups()
            # Use the combination of prefix and suffix as the key
            key = (prefix, suffix)
            # Add the feature to the corresponding group in the dictionary
            if key not in groups:
                groups[key] = []
            groups[key].append(feature)
            matched_features.add(feature)

    # Filter groups where the number of features is less than 4
    grouped_features = {k: v for k, v in groups.items() if len(v) >= 5}

    # Update matched features based on filtered groups
    updated_matched_features = set()
    for features_list in grouped_features.values():
        updated_matched_features.update(features_list)
    unmatched_features = [
        feature for feature in features if feature not in updated_matched_features
    ]
    return grouped_features, unmatched_features


def find_ohlc_groups(features):
    # Initialize an empty dictionary to store the groups
    pattern = re.compile(r"^(.*?)(open|high|low|close)(.*)$")

    # Initialize a dictionary to hold the matched groups
    groups = {}
    matched_features = set()

    # Iterate over the features to match and group them
    for feature in features:
        match = pattern.match(feature)
        if match:
            # Extract the base (prefix and suffix), keyword, and suffix
            prefix, keyword, suffix = match.groups()
            # Use the combination of prefix and suffix as the key
            key = (prefix, suffix)
            # Add the feature to the corresponding group in the dictionary
            if key not in groups:
                groups[key] = []
            groups[key].append(feature)
            matched_features.add(feature)

    # Filter groups where the number of features is less than 4
    grouped_features = {k: v for k, v in groups.items() if len(v) >= 4}

    # Update matched features based on filtered groups
    updated_matched_features = set()
    for features_list in grouped_features.values():
        updated_matched_features.update(features_list)

    unmatched_features = [
        feature for feature in features if feature not in updated_matched_features
    ]

    return grouped_features, unmatched_features


def find_strings_in_range(string_list, start_time, end_time):
    result = []

    start_time = datetime.strptime(start_time, "%Y-%m-%d")
    end_time = datetime.strptime(end_time, "%Y-%m-%d")
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    for string in string_list:
        matches = pattern.findall(string)
        for match in matches:
            try:
                time = datetime.strptime(match, "%Y-%m-%d")
                if start_time <= time <= end_time:
                    result.append(string)
            except ValueError:
                pass
    return result


def match_strings_in_range(string_list, single_date):
    result = []
    # 将传入的日期字符串转换为 date 对象
    try:
        date = datetime.strptime(single_date, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"format error for single date: {e}")
        return result  # 返回空列表

    # 用于找到日期的正则表达式
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

    for string in string_list:
        try:
            matches = pattern.findall(string)  # 找到所有日期匹配项
            for match in matches:
                # 将找到的字符串日期转换为 date 对象
                match_date = datetime.strptime(match, "%Y-%m-%d").date()
                # 如果日期匹配，添加到结果列表中
                if match_date == date:
                    result.append(string)
                    break  # 已找到匹配项，不需继续检查其他日期
        except ValueError as e:
            print(
                f"Could not find corresponding date when handling '{string}', the error is {e}"
            )
            # 我们可以选择跳过这个错误，继续处理列表中的下一个字符串
            continue
    return result[0]


if __name__ == "__main__":
    root_path = "./DOWNLOAD_DATASET/binance-futures/BTCUSDT/book_snapshot_25"
    file_list = os.listdir(root_path)
    result_1 = find_strings_in_range(file_list, "2023-01-01", "2023-01-05")
    result_2 = match_strings_in_range(file_list, "2023-01-01")
    print(result_1)
    print(result_2)
