import configparser, requests


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
        return json_response



