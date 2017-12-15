# Copyright 2017 Project PROFIT (http://projectprofit.eu/)
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.  You may
# obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# Licence for the specific language governing permissions and limitations
# under the Licence.

import logging, configparser, json, time, os, sys, datetime, atexit
sys.path.append('../../')
from flask import Flask, request, Response, abort
from logging.handlers import TimedRotatingFileHandler
from recommender_system.service.recommender import RecommenderSystem
from recommender_system.service.data_retriever import RatingApiHandler, ArticleSimilaritiesHandler
from flask import send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler

# global variables
CONFIG_FILE = 'config.ini'
LOGS_DIR = 'logs'
IMGS_DIR = 'img'
DATA_DIR = 'data'
RECOMMENDER_DATA_FILE = DATA_DIR + "/recommender_data.pkl"
ARTICLE_SIMILARITIES_FILE = DATA_DIR + "/sim_mx.pkl"
BLOCK_LIST = []  # fill this list with ips to be blocked (they will always get a 403 error)
SCHEDULER = BackgroundScheduler()  # variable to schedule tasks
app = Flask(__name__)


def main():
    create_folders()
    setup_logger()
    service_config = load_config(CONFIG_FILE)
    port = int(service_config['port'])

    # start scheduler and assign it to update recommender data
    SCHEDULER.start()
    schedule_recommender_data_update(SCHEDULER, service_config['update_data_hours'])
    atexit.register(lambda: SCHEDULER.shutdown())  # Shut down the scheduler when exiting the app

    app.run(host='0.0.0.0', port=port, threaded=True)


def create_folders():
    """
    create the necessary folders for the app to run
    :return:
    """
    # create log folder if it does not exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    # create data folder if it does not exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def setup_logger():
    """
    configure logging mechanism
    :return:
    """
    # logger used in flask, by default it prints only to stdout, we configure it to write to file as well
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.DEBUG)

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


def update_recommender_data():
    print('Recommender data update started at', datetime.datetime.now())
    recommender = RecommenderSystem()
    recommender.save_recommender_data(CONFIG_FILE, RECOMMENDER_DATA_FILE)


def schedule_recommender_data_update(scheduler, hours):
    scheduler.add_job(
        func=update_recommender_data,
        # trigger=IntervalTrigger(days=1, start_date='2010-10-10 11:25:00', timezone='EET'),
        trigger='cron',
        hour=hours,
        id='update_recommender_data',
        name='Update recommender data',
        replace_existing=True)


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
    # if set to true, all calculations are made real-time (very slow option)
    on_the_fly = params.get('on_the_fly') == 'true'

    print()
    print("ITEM RECOMMENDER SERVICE REQUEST")
    print("User id:", user_id)
    print("Max number of recommendations:", max_items)
    print("Method:", method)

    valid_methods = ['user', 'item', 'item_h', 'hybrid', 'content']
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
        if on_the_fly:
            recommender.calculate_info(CONFIG_FILE, method)
        else:
            # save recommender data if they do not exist, otherwise load them from file
            if not os.path.isfile(RECOMMENDER_DATA_FILE):
                recommender.save_recommender_data(CONFIG_FILE, RECOMMENDER_DATA_FILE)
            else:
                recommender = RecommenderSystem.load_recommender_data(RECOMMENDER_DATA_FILE)
        # check validity of user id based on retrieved data
        if recommender.is_user_id_valid(user_id):
            recommendations = recommender.recommend_items(user_id, max_items, method=method)
            response_object['input_user_id'] = user_id
            response_object['recommended_items'] = recommendations
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
    # if set to true, all calculations are made real-time (very slow option)
    on_the_fly = params.get('on_the_fly') == 'true'

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
        if on_the_fly:
            recommender.update_user_info(CONFIG_FILE)
            recommender.update_correlations_and_neighbourhood("user")
        else:
            # save recommender data if they do not exist
            if not os.path.isfile(RECOMMENDER_DATA_FILE):
                recommender.save_recommender_data(CONFIG_FILE, RECOMMENDER_DATA_FILE)
            else:
                recommender = RecommenderSystem.load_recommender_data(RECOMMENDER_DATA_FILE)
        # check validity of user id based on retrieved data
        if recommender.is_user_id_valid(user_id):
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


# @app.route('/update_sims')
# def update_sims():
#     ArticleSimilaritiesHandler.update_article_sims(CONFIG_FILE)
#     return ""


if __name__ == '__main__':
    main()

