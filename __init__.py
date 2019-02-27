# from langdetect import DetectorFactory
# from langdetect.lang_detect_exception import LangDetectException
# from langdetect.detector_factory import create_detector

# Testing..
from detector_factory import DetectorFactory, create_detector
from lang_detect_exception import LangDetectException

DetectorFactory.seed = 0
_detector = create_detector()


def langdetect_detect(col_val):
    _detector.reset()
    if not isinstance(col_val, str):
        raise LangDetectException(col_val, 'Not a string')
    _detector.append(col_val)
    result = _detector.detect()
    return result


def langdetect_cnn(col_val):
    from cnndetector import predict
    config = []
    res = predict(col_val, config, False)
    return res


def analyze_column_language(df, columns, detector_fn=langdetect_detect):
    """
    Goes over each column and row and tries to figure out it's language
    :param df: Dataframe to analyze
    :param columns: The columns to detect the language for
    :return: A dict of { language: occurances }
    """
    lang_stats = {
        'by_cols': {}
    }
    if isinstance(columns, str):
        columns = [columns]
    for index, row in df.iterrows():
        for column in columns:
            col_val = row[column]
            try:
                lang = detector_fn(col_val)
            except LangDetectException as e:
                lang = 'INV'
            df.loc[index, column + ".lng"] = lang
            if lang not in lang_stats:
                lang_stats[lang] = 0
            if column not in lang_stats['by_cols']:
                lang_stats['by_cols'][column] = {}
            if lang not in lang_stats:
                lang_stats[column][lang] = 0
            if lang not in lang_stats['by_cols'][column]:
                lang_stats['by_cols'][column][lang] = 0
            lang_stats[lang] += 1
            lang_stats['by_cols'][column][lang] += 1
    return lang_stats
