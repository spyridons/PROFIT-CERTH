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

import configparser, requests, pickle, os, datetime, subprocess


class RatingApiHandler:
    """
    Class responsible for retrieving rating data for the PROFIT project ('star_rating' items)
    """

    def __init__(self, configuration_file):
        self.token_url = None  # Url of the service to get authentication token
        self.ratings_base_url = None  # Url of the service to get ratings
        self.username = None  # username for the authentication token service
        self.password = None  # password for the authentication token service
        self.load_config(configuration_file)  # initialize the above variables

    def load_config(self, filename):
        """
        Set rating api variables from a configuration file
        :param filename: the configuration file
        :return:
        """
        config = configparser.ConfigParser()
        config.read(filename)
        rating_api_config = config['RATING_API']
        self.token_url = rating_api_config['token_url']
        self.ratings_base_url = rating_api_config['ratings_url']
        self.username = rating_api_config['username']
        self.password = rating_api_config['password']

    def get_authentication_token(self):
        """
        get authentication token for the ratings api
        :return: the token string
        """

        # set data and headers, send request and return the token response
        data = {'username': self.username, 'password': self.password}
        headers = {'Accept': 'application/json'}
        response = requests.post(self.token_url, data=data, headers=headers)
        json_response = response.json()
        token = json_response['token']
        return token

    def get_ratings(self, token, timestamp=None):
        """
        get ratings from api
        :param token: the authentication token from the request
        :param timestamp: optional, if we want ratings after a specified time
        :return: ratings in json format
        """

        # form request headers
        token_field = 'Token ' + token
        headers = {'Accept': 'application/json', 'Authorization': token_field}

        # add timestamp parameter is requested
        ratings_url = self.ratings_base_url
        if timestamp is not None:
            ratings_url += "?timestamp_from=" + timestamp

        # send request and return response
        response = requests.get(ratings_url, headers=headers)
        json_response = response.json()
        # json_response = json.load(open('data.json'))
        return json_response


class ArticleSimilaritiesHandler:

    @staticmethod
    def update_article_sims(config_file):
        """
        function can be removed when the executions will be run automatically
        :param config_file:
        :return:
        """
        config = configparser.ConfigParser()
        config.read(config_file)
        swc_api_config = config['SWC_API']
        similarities_file = swc_api_config['article_similarities_path']
        scripts_dir = swc_api_config['scripts_dir']
        update_articles_script = swc_api_config['update_articles_script']
        compute_similarities_script = swc_api_config['compute_similarities_script']
        date_m_epoch = os.path.getmtime(similarities_file)
        date_m = datetime.datetime.utcfromtimestamp(date_m_epoch)
        date = (date_m - datetime.timedelta(hours=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
        subprocess.run("python " + compute_similarities_script + " " + date, cwd=scripts_dir)

    @staticmethod
    def get_article_similarities(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        swc_api_config = config['SWC_API']
        similarities_file = swc_api_config['article_similarities_path']
        api_result = pickle.load(open(similarities_file, "rb"))
        uri_id_dict = api_result[1]
        article_similarities_matrix = api_result[2]
        return uri_id_dict, article_similarities_matrix
