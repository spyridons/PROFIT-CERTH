import logging, configparser, json, time, os, sys
sys.path.append('../../')
from flask import Flask, request, Response, abort
from logging.handlers import TimedRotatingFileHandler
from recommender_system.service.recommender import RecommenderSystem
from recommender_system.service.data_retriever import RatingApiHandler
from flask import send_from_directory

# global variables
CONFIG_FILE = 'config.ini'
LOGS_DIR = 'logs'
IMGS_DIR = 'img'
BLOCK_LIST = []  # fill this list with ips to be blocked (they will always get a 403 error)
app = Flask(__name__)


def main():
    setup_logger()
    service_config = load_config(CONFIG_FILE)
    port = service_config['port']
    app.run(host='0.0.0.0', port=port, threaded=True)


def setup_logger():
    """
    configure logging mechanism
    :return:
    """
    # logger used in flask, by default it prints only to stdout, we configure it to write to file as well
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.DEBUG)

    # create log folder if it does not exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    file_handler = TimedRotatingFileHandler(LOGS_DIR + '/log', when='D')
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_config(filename):
    """
    Load service-related configurations from file
    :param filename:
    :return:
    """
    config = configparser.ConfigParser()
    config.read(filename)
    service_config = config['SERVICE']
    return service_config


@app.before_request
def check_ip():
    """
    block unwanted ips
    :return:
    """
    if request.remote_addr in BLOCK_LIST:
        abort(403)


@app.route('/recommend_items', methods=['GET'])
def recommend_items():
    """
    retrieve item recommendations
    input parameters:
    - user_id, which user should get the recommendations
    - max_num, maximum number of recommendations
    - method, 'user' for user-based recommendations or 'item' for item-based
    :return:
    """
    start = time.time()

    params = request.args
    user_id = params.get('user_id', type=int)
    max_items = params.get('max_num', type=int)
    method = params.get('method', 'user')

    print()
    print("ITEM RECOMMENDER SERVICE REQUEST")
    print("User id:", user_id)
    print("Max number of recommendations:", max_items)
    print("Method:", method)

    valid_methods = ['user', 'item']
    status_code = 200

    response_object = dict()
    if user_id is None or max_items is None:
        status_code = 400
        error = 'Invalid parameters (user_id or max_num)'
        response_object['error'] = error
    elif method not in valid_methods:
        status_code = 400
        error = 'Invalid recommendation method'
        response_object['error'] = error
    else:
        recommender = RecommenderSystem()
        recommender.update_user_info(CONFIG_FILE)
        # check validity of user id based on retrieved data
        if recommender.is_user_id_valid(user_id):
            recommender.update_neighbourhood(method)
            recommendations = recommender.recommend_items(user_id, max_items, method=method)
            response_object['input_user_id'] = user_id
            response_object['recommended_item_ids'] = recommendations
        else:
            status_code = 400
            error = 'user id must have values between 1 and ' + str(recommender.max_user_id())
            response_object['error'] = error
    response = Response(response=json.dumps(response_object),
                        status=status_code,
                        mimetype="application/json")
    elapsed = time.time() - start
    print("Recommender service elapsed time:", elapsed, "seconds")
    return response


@app.route('/recommend_users', methods=['GET'])
def recommend_users():
    """
    retrieve user recommendations
    input parameters:
    - user_id, which user should get the recommendations
    - max_num, maximum number of recommendations
    :return:
    """
    start = time.time()

    params = request.args
    user_id = params.get('user_id', type=int)
    max_users = params.get('max_num', type=int)

    print()
    print("USER RECOMMENDER SERVICE REQUEST")
    print("User id:", user_id)
    print("Max number of recommendations:", max_users)

    status_code = 200

    response_object = dict()
    if user_id is None or max_users is None:
        status_code = 400
        error = 'Invalid parameters (user_id or max_num)'
        response_object['error'] = error
    else:
        recommender = RecommenderSystem()
        recommender.update_user_info(CONFIG_FILE)
        # check validity of user id based on retrieved data
        if recommender.is_user_id_valid(user_id):
            recommender.update_neighbourhood("user")
            recommendations = recommender.recommend_users(user_id, max_users)
            response_object['input_user_id'] = user_id
            response_object['recommended_user_ids'] = recommendations
        else:
            status_code = 400
            error = 'user id must have values between 1 and ' + str(recommender.max_user_id())
            response_object['error'] = error
    response = Response(response=json.dumps(response_object),
                        status=status_code,
                        mimetype="application/json")
    elapsed = time.time() - start
    print("Recommender service elapsed time:", elapsed, "seconds")
    return response


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, IMGS_DIR),
                               'favicon.png', mimetype='image/vnd.microsoft.icon')


# @app.route('/ratings')
def ratings():
    """
    For debugging purposes, return the ratings info from the respective API
    :return:
    """
    rating_api_handler = RatingApiHandler(CONFIG_FILE)
    auth_token = rating_api_handler.get_authentication_token()
    ratings_json = rating_api_handler.get_ratings(auth_token)
    response = Response(response=json.dumps(ratings_json),
                        status=200,
                        mimetype="application/json")
    return response


if __name__ == '__main__':
    main()

