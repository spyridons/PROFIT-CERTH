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

from scipy.sparse import csr_matrix
from enum import Enum
import logging
import numpy as np
import warnings, sys, pickle, time
sys.path.append('../../')
from recommender_system.service.data_retriever import RatingApiHandler, ArticleSimilaritiesHandler
from random import shuffle

LOGGER = logging.getLogger('werkzeug')


class RatingMode(Enum):
    RATING_ONLY = "Use only the initial rating for the final score."
    RATING_USING_MEANS = "Use the initial rating minus the mean rating score."
    RATING_USING_MEANS_AND_STD = "Use the initial rating minus the mean rating score. " \
                                 "Divide the difference by the standard deviation of the ratings."


class RecommenderSystem:
    SAVING_PROCESS_ACTIVE = False
    RATING_MODE = RatingMode.RATING_USING_MEANS
    HYBRID_WEIGHT_U = 0.4
    HYBRID_WEIGHT_I = 0.2
    HYBRID_WEIGHT_C = 0.4
    DEBUG = False

    def __init__(self):
        self.ui_matrix = None  # user-article matrix (values are the rating scores)
        self.ii_matrix = None  # article-article similarity
        self.users_correlations = None
        self.items_correlations = None
        self.users_neighbourhood = None
        self.items_neighbourhood = None

        # content got by SWC
        self.items_neighbourhood_ii = None
        self.uri_id_dict = None
        self.id_uri_dict = None

        self.default_recommendations = None
        self.default_user_recommendations = None

        self.err_msg = ''

        self.user_blacklist = []
        self.item_blacklist = []

    def add_item_to_blacklist(self, uri):
        # add id into the black list, not the uri
        if uri in self.uri_id_dict.keys():
            id = self.uri_id_dict[uri]
            self.item_blacklist.append(id)

    def add_user_to_blacklist(self, id):
        self.user_blacklist.append(id)

    def is_user_id_valid(self, user_id):
        """
        used for validating input user id
        :param user_id:
        :return:
        """
        if self.ui_matrix is not None:
            if 1 <= user_id <= self.ui_matrix.shape[0]:
                return True
        return False

    def max_user_id(self):
        if self.ui_matrix is None:
            return -1
        else:
            return self.ui_matrix.shape[0]

    def calculate_and_save_recommender_data(self, config_file, recommender_data_file):
        """
        calculate and store ui_matrix, ii_matrix, correlations and neighbourhoods to a pickle file
        :param config_file:
        :param recommender_data_file:
        :return:
        """

        # calculate recommender data (ui_matrix, ii_matrix, correlations and neighbourhoods)
        LOGGER.debug("Calculating recommender data...")
        self.update_item_info(config_file)
        self.update_user_info(config_file)
        self.update_correlations_and_neighbourhood('user')
        self.update_correlations_and_neighbourhood('item')
        self.create_neighbourhood_dict('item-ii')
        self.create_default_recommendations()
        self.create_default_user_recommendations()

        # shrink large arrays
        self.items_correlations = RecommenderSystem.shrink_array(self.items_correlations)
        self.users_correlations = RecommenderSystem.shrink_array(self.users_correlations)

        self.save_recommender_data(recommender_data_file)

    def save_recommender_data(self, recommender_data_file):
        # store the recommender data to a pickle file
        LOGGER.debug("Storing recommender data...")
        complete = False
        while not complete:
            if RecommenderSystem.SAVING_PROCESS_ACTIVE:
                LOGGER.warning("Saving stalled for 0.5 seconds... another saving process is being executed right now!!!")
                time.sleep(0.5)
            else:
                RecommenderSystem.SAVING_PROCESS_ACTIVE = True
                pickle.dump(self, open(recommender_data_file, "wb"))
                RecommenderSystem.SAVING_PROCESS_ACTIVE = False
                complete = True
        LOGGER.debug("Finished storing recommender data...")

    @staticmethod
    def load_recommender_data(recommender_data_file):
        """
        load a RecommenderSystem object from a pickle file
        and set the appropriate fields
        :param recommender_data_file:
        :return:
        """

        loaded = False
        while not loaded:
            if RecommenderSystem.SAVING_PROCESS_ACTIVE:
                LOGGER.warning("Loading stalled for 0.5 seconds... saving process is being executed right now!!!")
                time.sleep(0.5)
            else:
                recommender_data = pickle.load(open(recommender_data_file, "rb"))
                loaded = True
        return recommender_data

    def calculate_info(self, config_file, method='user'):
        """
        for on-the-fly implementation
        :param method:
        :return:
        """
        self.update_item_info(config_file)
        self.update_user_info(config_file)
        self.create_default_recommendations()
        if method == 'user':
            self.update_correlations_and_neighbourhood('user')
        elif method == 'item':
            self.update_correlations_and_neighbourhood('item')
        elif method == 'item_h':
            self.create_neighbourhood_dict('item-ii')
        elif method == 'hybrid':
            self.update_correlations_and_neighbourhood('user')
            self.update_correlations_and_neighbourhood('item')

    def update_user_info(self, config_file):
        """
        function to store user-item matrix
        :return:
        """
        rating_api_handler = RatingApiHandler(config_file)
        auth_token = rating_api_handler.get_authentication_token()
        ratings_json = rating_api_handler.get_ratings(auth_token)
        self.create_ui_matrix(ratings_json)

    def update_item_info(self, config_file):
        """
        function to store item-item matrix
        :param config_file:
        :return:
        """
        uri_id_dict, article_similarities_matrix = ArticleSimilaritiesHandler.get_article_similarities(config_file)
        self.uri_id_dict = uri_id_dict
        self.id_uri_dict = {v: k for k, v in uri_id_dict.items()}
        self.ii_matrix = article_similarities_matrix

    def create_ui_matrix(self, ratings_json):
        """
        Note that user and item indices are decremented by 1 in order to start from zero
        :param ratings_json:
        :return:
        """
        values = list()
        row_idxs = list()
        col_idxs = list()
        max_user_id = max(set(rating['rated_by'] for rating in ratings_json))
        # max_item_id = max(set(rating['object_id'] for rating in ratings_json))
        max_item_id = max(self.uri_id_dict.values())
        for rating in ratings_json:
            user_id = rating['rated_by']
            item_uri = rating['uri']
            if item_uri is None:
                continue
            if item_uri in self.uri_id_dict.keys():
                item_id = self.uri_id_dict[item_uri]
                rating = rating['rating']
                if item_id <= max_item_id:
                    values.append(rating)
                    user_id_in_ui = user_id - 1
                    row_idxs.append(user_id_in_ui)
                    # col_idxs.append(item_id - 1)
                    col_idxs.append(item_id)
            else:
                max_item_id += 1
                item_id = max_item_id
                self.id_uri_dict[item_id] = item_uri
                self.uri_id_dict[item_uri] = item_id
                rating = rating['rating']

                values.append(rating)
                user_id_in_ui = user_id - 1
                row_idxs.append(user_id_in_ui)
                # col_idxs.append(item_id - 1)
                col_idxs.append(item_id)

                # update ii matrix (add one empty row and column), and uri-id dictionaries
                ii_matrix_shape = self.ii_matrix.shape
                self.ii_matrix = csr_matrix((self.ii_matrix.data, self.ii_matrix.indices,
                                             np.hstack((self.ii_matrix.indptr, self.ii_matrix.indptr[-1]))),
                                            shape=(ii_matrix_shape[0] + 1, ii_matrix_shape[1] + 1))
                # self.ii_matrix.reshape(shape=(ii_matrix_shape[0] + 1, ii_matrix_shape[1] + 1))
                # self.id_uri_dict = item_id
        columns = max_item_id + 1
        self.ui_matrix = csr_matrix((values, (row_idxs, col_idxs)), shape=(max_user_id, columns))

    def update_correlations_and_neighbourhood(self, unit='user'):
        """
        store correlations and neighbourhood
        :param unit:
        :return:
        """
        self.form_pearson_correlation_matrix(unit)
        self.create_neighbourhood_dict(unit)

    def form_pearson_correlation_matrix(self, unit='user', step=1000):
        """
        Get pearson correlations (for users or items)
        :param unit:
        :param step:
        :return:
        """

        # define the matrix from which we will extract correlations
        if unit == 'user':
            if self.ui_matrix is None:
                LOGGER.error("User-item matrix does not exist, can't create correlations!!!")
                return
            target_matrix = self.ui_matrix.copy()
            num_rows = target_matrix.shape[0]
            correlation_matrix = self.users_correlations = np.zeros(shape=(num_rows, num_rows))
        elif unit == 'item':
            if self.ui_matrix is None:
                LOGGER.error("User-item matrix does not exist, can't create correlations!!!")
                return
            target_matrix = self.ui_matrix.transpose().tocsr().copy()
            num_rows = target_matrix.shape[0]
            correlation_matrix = self.items_correlations = np.zeros(shape=(num_rows, num_rows))
        else:
            LOGGER.error("Wrong matrix parameter value!!!")
            return

        # iterate though rows of the matrix and calculate correlation in batches
        for col_offset in range(0, num_rows, step):
            col_step = step
            # for end of loop, where we may need less columns
            if (col_offset + step) > num_rows:
                col_step = num_rows - col_offset
            for row_offset in range(col_offset, num_rows, step):
                # print("Rows:", row_offset)
                # print("Columns:", col_offset)
                row_step = step
                # for end of loop, where we may need less rows
                if (row_offset + step) > num_rows:
                    row_step = num_rows - row_offset
                first_input_param = target_matrix[col_offset:col_offset + col_step, :].todense()
                second_input_param = target_matrix[row_offset:row_offset + row_step, :].todense()
                # ignore warnings regarding zero std values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    correlations = np.corrcoef(first_input_param, second_input_param)
                # Resulting matrix appends the vectors of the second input parameter
                # below the vectors of the first input parameter.
                # We want the correlations of the vectors of the second with the vectors in the first.
                # So, desired info is in last rows and in first columns, depending on the step
                to_append = correlations[-row_step:, :col_step]
                correlation_matrix[row_offset:row_offset + row_step, col_offset:col_offset + col_step] = to_append
                # get the symmetric slice and append it as well to avoid extra calculations
                if col_offset != row_offset:
                    to_append2 = correlations[:col_step, -row_step:]
                    correlation_matrix[col_offset:col_offset + col_step, row_offset:row_offset + row_step] = to_append2
        return correlation_matrix

    def create_neighbourhood_dict(self, unit='user'):
        """
        Create a dictionary containing the neighbours of each unit (user or item)
        Indices start from 0 like in the ui matrix (instead of 1).
        :return:
        """

        if unit == 'user':
            correlation_matrix = self.users_correlations
            if self.users_correlations is None:
                LOGGER.error("User correlations do not exist, can't create neighbourhood!!!")
                return
            neighbourhood = self.users_neighbourhood = dict()
        elif unit == 'item':
            correlation_matrix = self.items_correlations
            if self.items_correlations is None:
                LOGGER.error("Item correlations do not exist, can't create neighbourhood!!!")
                return
            neighbourhood = self.items_neighbourhood = dict()
        elif unit == 'item-ii':
            # in order to consider all cases as a numpy array
            correlation_matrix = self.ii_matrix.copy().toarray()
            if self.ii_matrix is None:
                LOGGER.error("Item similarities do not exist, can't create neighbourhood!!!")
                return
            neighbourhood = self.items_neighbourhood_ii = dict()
        else:
            LOGGER.error("Wrong matrix parameter value!!!")
            return

        for id in range(len(correlation_matrix)):
            # for id in range(correlation_matrix.shape[0]):
            correlation_vector = correlation_matrix[id, :]
            correlation_vector = [x if not np.isnan(x) else 0 for x in correlation_vector]
            sorted_indices = sorted(range(len(correlation_vector)), key=lambda i: correlation_vector[i])
            sorted_indices.remove(id)
            sorted_indices.reverse()
            if len(sorted_indices) > 500:
                sorted_indices = sorted_indices[:500]
            neighbourhood[id] = sorted_indices

    def create_default_recommendations(self):
        mean_scores = []
        num_ratings_list = []
        ui_matrix_csc = self.ui_matrix.copy().tocsc()
        for i in range(ui_matrix_csc.shape[1]):
            col = ui_matrix_csc.getcol(i)
            num_ratings = col.nnz
            if num_ratings == 0:
                mean_scores.append(0)
            else:
                mean_score = col.sum() / num_ratings
                mean_scores.append(mean_score)
            num_ratings_list.append(num_ratings)
        # sort based on two criteria, first is the mean rating score, second is num users who rated the item
        sorted_indices = sorted(range(len(mean_scores)), key=lambda i: (mean_scores[i], num_ratings_list[i]))
        sorted_indices.reverse()
        self.default_recommendations = sorted_indices

    def create_default_user_recommendations(self):
        num_ratings_list = []
        for i in range(self.ui_matrix.shape[0]):
            row = self.ui_matrix.getrow(i)
            num_ratings = row.nnz
            num_ratings_list.append(num_ratings)
        # sort based on the number of ratings
        sorted_indices = sorted(range(len(num_ratings_list)), key=lambda i: num_ratings_list[i])
        sorted_indices.reverse()
        self.default_user_recommendations = sorted_indices

    def recommend_items(self, user_id, max_recommendations, method='user'):
        if method == 'hybrid':
            predictions_u = self.get_predictions(user_id, method='user')
            predictions_u = [x if 0 <= x <= 5 else 5.0 if x >= 5 else 0.0 for x in predictions_u]
            predictions_i = self.get_predictions(user_id, method='item')
            predictions_i = [x if 0 <= x <= 5 else 5.0 if x >= 5 else 0.0 for x in predictions_i]
            predictions_c = self.get_predictions(user_id, method='content')
            predictions_c = [x if 0 <= x <= 5 else 5.0 if x >= 5 else 0.0 for x in predictions_c]
            if len(predictions_u) == 0 or len(predictions_i) == 0 or len(predictions_c) == 0:
                return [self.err_msg]
            predictions = [RecommenderSystem.HYBRID_WEIGHT_U * x_u +
                           RecommenderSystem.HYBRID_WEIGHT_I * x_i +
                           RecommenderSystem.HYBRID_WEIGHT_C * x_c
                           for x_u, x_i, x_c in zip(predictions_u, predictions_i, predictions_c)]

            # ONLY FOR DEBUGGING
            if self.DEBUG:
                top_n = 60
                if all([p == 0 for p in predictions]):
                    LOGGER.debug("Default recommendations:")
                    for x in self.default_recommendations[:top_n]:
                        LOGGER.debug(x, self.id_uri_dict[x])
                else:
                    self.print_top_predictions(predictions_u, predictions_i, predictions_c, predictions, top_n)
        else:
            predictions = self.get_predictions(user_id, method=method)
            if len(predictions) == 0:
                return [self.err_msg]

        if all([p == 0 for p in predictions]):
            recommendations = self.recommend_default(max_recommendations)
        else:
            sorted_indices = sorted(range(len(predictions)), key=lambda i: predictions[i])
            sorted_indices.reverse()

            # remove deleted items
            sorted_indices = [x for x in sorted_indices if x not in self.item_blacklist]

            recommendations = [self.id_uri_dict[x] for x in sorted_indices[:max_recommendations]]
        return recommendations

    def recommend_default(self, max_recommendations):
        # remove deleted items
        self.default_recommendations = [x for x in self.default_recommendations if x not in self.item_blacklist]

        return [self.id_uri_dict[x] for x in self.default_recommendations[:max_recommendations]]

    def recommend_default_users(self, max_recommendations):
        # remove deleted users
        self.default_user_recommendations = [x for x in self.default_user_recommendations if x not in self.user_blacklist]

        return [x for x in self.default_user_recommendations[:max_recommendations]]

    def get_predictions(self, user_id, max_neighbours=10, method='user'):
        """
        Recommend items for the given user id
        :param user_id:
        :param max_recommendations:
        :param max_neighbours:
        :param method: 'user' for user-based recommendations, 'item' for item-based
        :return:
        """

        if method == 'content':
            # These variables must not be None
            to_check = [self.ui_matrix, self.ii_matrix]
            to_check_labels = ['UI matrix', 'II matrix']
        else:
            if method == 'user':
                correlations = self.users_correlations
                neighbourhoods = self.users_neighbourhood
                to_check_labels = ['UI matrix', 'users correlations', 'users neighbourhood']
            elif method == 'item':
                correlations = self.items_correlations
                neighbourhoods = self.items_neighbourhood
                to_check_labels = ['UI matrix', 'items correlations', 'items neighbourhood']
            elif method == 'item_h':
                # content-based correlations and neighbourhoods
                correlations = self.ii_matrix
                neighbourhoods = self.items_neighbourhood_ii
                to_check_labels = ['UI matrix', 'II matrix', 'items neighbourhood']

            # These variables must not be None
            to_check = [self.ui_matrix, correlations, neighbourhoods]

        for idx, val in enumerate(to_check):
            if val is None:
                self.err_msg = "Error: Missing " + to_check_labels[idx]
                return []

        user_id_in_ui = user_id - 1
        # for each zero rating, predict ratings
        user_rating_row = self.ui_matrix.getrow(user_id_in_ui)
        zero_idxs = np.where(user_rating_row.toarray() == 0)[1]
        predictions = np.zeros(shape=self.ui_matrix.shape[1])

        # prepare some vars to speed up content-based recommendations
        if method == 'content':
            num_rated_items = user_rating_row.nnz
            values = user_rating_row.data
            columns = user_rating_row.indices
            user_rating_row_pairs = list(zip(values, columns))
            shuffle(user_rating_row_pairs)
            sorted_user_rating_row_pairs = sorted(user_rating_row_pairs, key=lambda pair: pair[0])
            sorted_user_rating_row_pairs.reverse()
            sorted_user_rating_row_pairs = sorted_user_rating_row_pairs[:10]

        # ONLY FOR DEBUGGING
        # print top similar items
        if self.DEBUG:
            if method == 'content':
                top_n = 60
                self.print_non_zero(user_rating_row, top_n)

        if user_rating_row.nnz == 0:
            return predictions
        for zero_idx in zero_idxs:
            # content based approach
            if method == 'content':

                # rows, cols = user_rating_row.nonzero()
                # sum_products = 0
                # for row, col in zip(rows, cols):
                #     item_idx = col
                #     rating = user_rating_row[0, item_idx]
                #     similarity = self.ii_matrix[zero_idx, item_idx]
                #     product = rating * similarity
                #     sum_products += product

                # speeded up version of the above commented-out lines
                # since 'user_rating_row' consists of only one row, '.indices' are the actual indices of the '.data'
                sum_products = 0
                for val, col in sorted_user_rating_row_pairs:
                    item_idx = col
                    similarity = self.ii_matrix[zero_idx, item_idx]
                    product = val * similarity
                    sum_products += product
                prediction = sum_products / num_rated_items
                predictions[zero_idx] = prediction

            # collaborative filtering approaches
            else:
                num_neighbours = 0
                correlations_sum = 0
                numerator = 0
                ratings_for_alpha_beta = None
                # discriminate user-based recommendations from item-based ones
                if method == 'user':
                    # get the neighbours of the user which will get recommendations
                    # take into account only user neighbours who rated the item
                    non_zero_neighbours = self.ui_matrix.getcol(zero_idx).nonzero()[0]
                    neighbourhood = neighbourhoods[user_id_in_ui]
                    neighbourhood = [x for x in neighbourhood if x in non_zero_neighbours]
                    for neighbour_id in neighbourhood:
                        # row = user which will get recommendations, column = the neighbour of the user
                        correlation = correlations[user_id_in_ui, neighbour_id]
                        if np.isnan(correlation):
                            correlation = 0
                        correlations_sum += abs(correlation)
                        neighbour_rating = self.ui_matrix[neighbour_id, zero_idx]

                        # in cases when we do not simply utilize the rating score
                        if RatingMode != RatingMode.RATING_ONLY:
                            neighbour_rating_row = self.ui_matrix.getrow(neighbour_id)
                            neighbour_rating = self.rating_function(neighbour_rating, neighbour_rating_row)

                        numerator += correlation * neighbour_rating
                        num_neighbours += 1
                        if num_neighbours == max_neighbours:
                            break
                    ratings_for_alpha_beta = self.ui_matrix.getrow(user_id_in_ui)
                elif method == 'item' or method == 'item_h':
                    # get the neighbours of the item to be rated
                    # take into account only neighbours rated by the user
                    non_zero_neighbours = self.ui_matrix.getrow(user_id_in_ui).nonzero()[1]
                    # consider only the first N neighbours for efficiency
                    neighbourhood = neighbourhoods[zero_idx][:200]
                    neighbourhood = [x for x in neighbourhood if x in non_zero_neighbours]
                    for neighbour_id in neighbourhood:
                        # row = the item to be rated, column = the neighbour of the item to be rated
                        correlation = correlations[zero_idx, neighbour_id]
                        if np.isnan(correlation):
                            correlation = 0
                        correlations_sum += abs(correlation)
                        neighbour_rating = self.ui_matrix[user_id_in_ui, neighbour_id]

                        # in cases when we do not simply utilize the rating score
                        if self.RATING_MODE != RatingMode.RATING_ONLY:
                            neighbour_rating_column = self.ui_matrix.getcol(neighbour_id)
                            neighbour_rating = self.rating_function(neighbour_rating, neighbour_rating_column)

                        numerator += correlation * neighbour_rating
                        num_neighbours += 1
                        if num_neighbours == max_neighbours:
                            break
                    ratings_for_alpha_beta = self.ui_matrix.getcol(zero_idx)

                denominator = correlations_sum if correlations_sum != 0 else 1
                alpha, beta = self.get_alpha_beta(ratings_for_alpha_beta)
                prediction = alpha + beta * numerator / denominator
                predictions[zero_idx] = prediction

        return predictions

    def rating_function(self, neighbour_rating, neighbour_total_ratings):
        """
        Get rating function score (used when we do not simply consider the rating score of a neighbour)
        :param neighbour_rating:
        :param neighbour_total_ratings: used to get mean and std for all the neighbour ratings
        :return:
        """
        rating_function_score = neighbour_rating
        mean_rating = neighbour_total_ratings.sum() / neighbour_total_ratings.nnz
        if self.RATING_MODE == RatingMode.RATING_USING_MEANS:
            rating_function_score = rating_function_score - mean_rating
        elif self.RATING_MODE == RatingMode.RATING_USING_MEANS_AND_STD:
            std_rating = np.std(neighbour_total_ratings.data)
            rating_function_score = (rating_function_score - mean_rating) / std_rating
        return rating_function_score

    def get_alpha_beta(self, ratings):
        """
        alpha and beta values for the predictions equation
        :param ratings:
        :return:
        """
        if self.RATING_MODE == RatingMode.RATING_ONLY or ratings is None:
            alpha = 0
            beta = 1
        elif self.RATING_MODE == RatingMode.RATING_USING_MEANS:
            if ratings.nnz == 0:
                alpha = 0
            else:
                alpha = ratings.sum() / ratings.nnz
            beta = 1
        elif self.RATING_MODE == RatingMode.RATING_USING_MEANS_AND_STD:
            if ratings.nnz == 0:
                alpha = 0
            else:
                alpha = ratings.sum() / ratings.nnz
            beta = np.std(ratings.data)
        return alpha, beta

    def recommend_users(self, user_id, max_recommendations):
        """
        Recommend users for the given user id
        :param user_id:
        :param max_recommendations:
        :return:
        """

        if self.users_neighbourhood is None:
            return ["Error: Missing user neighbourhood"]
        else:
            user_id_in_ui = user_id - 1
            neighbours = self.users_neighbourhood[user_id_in_ui]

            # remove deleted users
            neighbours = [x for x in neighbours if x not in self.user_blacklist]

            # increment by one to return the original indices
            recommendations = [x + 1 for x in neighbours[:max_recommendations]]
            return recommendations


    @staticmethod
    def shrink_array(matrix):
        """
        Returns a csr matrix by keeping only the values above zero
        :param matrix: a numpy array
        :return:
        """

        matrix[np.isnan(matrix)] = -1
        matrix[matrix < 0] = 0
        matrix = csr_matrix(matrix)
        matrix.eliminate_zeros()
        matrix = csr_matrix.maximum(matrix, matrix.transpose())

        return matrix

    # DEBUGGING FUNCTIONS BELOW!!!
    def print_top_predictions(self, predictions_u, predictions_i, predictions_c, predictions, top_n):
        print("Top predictions user-based:")
        print('id uri prediction_user prediction_item prediction_content final_prediction')
        top_predictions_u = sorted(range(len(predictions_u)), key=lambda i: predictions_u[i])
        top_predictions_u.reverse()
        for i in range(top_n):
            top_prediction = top_predictions_u[i]
            top_prediction_uri = self.id_uri_dict[top_prediction]
            print(top_prediction, top_prediction_uri,
                  predictions_u[top_prediction], predictions_i[top_prediction], predictions_c[top_prediction],
                  predictions[top_prediction])
        print()

        print("Top predictions item-based:")
        print('id uri prediction_user prediction_item prediction_content final_prediction')
        top_predictions_i = sorted(range(len(predictions_i)), key=lambda i: predictions_i[i])
        top_predictions_i.reverse()
        for i in range(top_n):
            top_prediction = top_predictions_i[i]
            top_prediction_uri = self.id_uri_dict[top_prediction]
            print(top_prediction, top_prediction_uri,
                  predictions_u[top_prediction], predictions_i[top_prediction], predictions_c[top_prediction],
                  predictions[top_prediction])
        print()

        print("Top predictions content-based:")
        print('id uri prediction_user prediction_item prediction_content final_prediction')
        top_predictions_c = sorted(range(len(predictions_c)), key=lambda i: predictions_c[i])
        top_predictions_c.reverse()
        for i in range(top_n):
            top_prediction = top_predictions_c[i]
            top_prediction_uri = self.id_uri_dict[top_prediction]
            print(top_prediction, top_prediction_uri,
                  predictions_u[top_prediction], predictions_i[top_prediction], predictions_c[top_prediction],
                  predictions[top_prediction])
        print()

        print("Top predictions final:")
        print('id uri prediction_user prediction_item prediction_content final_prediction')
        top_predictions = sorted(range(len(predictions)), key=lambda i: predictions[i])
        top_predictions.reverse()
        for i in range(top_n):
            top_prediction = top_predictions[i]
            top_prediction_uri = self.id_uri_dict[top_prediction]
            print(top_prediction, top_prediction_uri,
                  predictions_u[top_prediction], predictions_i[top_prediction], predictions_c[top_prediction],
                  predictions[top_prediction])
        print()

    def print_non_zero(self, user_rating_row, top_n):
        non_zero_idxs = user_rating_row.nonzero()[1]
        for non_zero_idx in non_zero_idxs:
            self.print_most_similar_items(non_zero_idx, top_n)

    def print_most_similar_items(self, item_id, num):
        print('Most similar items for id', item_id, ":", self.id_uri_dict[item_id])
        print('id uri similarity')
        item_row = self.ii_matrix.getrow(item_id).toarray()[0]
        top_article_ids = sorted(range(len(item_row)), key=lambda i: item_row[i])
        top_article_ids.reverse()
        for i in range(num):
            print(top_article_ids[i], self.id_uri_dict[top_article_ids[i]], self.ii_matrix[item_id, top_article_ids[i]])
        print()
