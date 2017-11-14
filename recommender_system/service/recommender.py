from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
import warnings, sys
sys.path.append('../../')
from recommender_system.service.data_retriever import RatingApiHandler


class RecommenderSystem:

    def __init__(self):
        self.ui_matrix = None  # user-article matrix (values are the rating scores)
        self.ii_matrix = None  # article-article similarity
        self.users_correlations = None
        self.items_correlations = None
        self.users_neighbourhood = None
        self.items_neighbourhood = None

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

    def update_user_info(self, config_file):
        """
        function to store user-item matrix
        :return:
        """
        rating_api_handler = RatingApiHandler(config_file)
        auth_token = rating_api_handler.get_authentication_token()
        ratings_json = rating_api_handler.get_ratings(auth_token)
        self.create_ui_matrix(ratings_json)

    def update_neighbourhood(self, unit='user'):
        """
        store correlations and neighbourhood
        :param unit:
        :return:
        """
        self.form_pearson_correlation_matrix(unit)
        self.create_neighbourhood_dict(unit)

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
        max_item_id = max(set(rating['object_id'] for rating in ratings_json))
        for rating in ratings_json:
            user_id = rating['rated_by']
            item_id = rating['object_id']
            rating = rating['rating']
            values.append(rating)
            row_idxs.append(user_id - 1)
            col_idxs.append(item_id - 1)
        self.ui_matrix = csr_matrix((values, (row_idxs, col_idxs)), shape=(max_user_id, max_item_id))

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
                print("User-item matrix does not exist, can't create correlations!!!")
                return
            target_matrix = self.ui_matrix.copy()
            num_rows = target_matrix.shape[0]
            correlation_matrix = self.users_correlations = np.zeros(shape=(num_rows, num_rows))
        elif unit == 'item':
            if self.ui_matrix is None:
                print("User-item matrix does not exist, can't create correlations!!!")
                return
            target_matrix = self.ui_matrix.transpose().tocsr().copy()
            num_rows = target_matrix.shape[0]
            correlation_matrix = self.items_correlations = np.zeros(shape=(num_rows, num_rows))
        else:
            print("Wrong matrix parameter value!!!")
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
                print("User correlations do not exist, can't create neighbourhood!!!")
                return
            neighbourhood = self.users_neighbourhood = dict()
        elif unit == 'item':
            correlation_matrix = self.items_correlations
            if self.items_correlations is None:
                print("Item correlations do not exist, can't create neighbourhood!!!")
                return
            neighbourhood = self.items_neighbourhood = dict()
        else:
            print("Wrong matrix parameter value!!!")
            return

        for id in range(len(correlation_matrix)):
            correlation_vector = correlation_matrix[id, :]
            correlation_vector = [x if not np.isnan(x) else 0 for x in correlation_vector]
            sorted_indices = sorted(range(len(correlation_vector)), key=lambda i: correlation_vector[i])
            sorted_indices.remove(id)
            sorted_indices.reverse()
            neighbourhood[id] = sorted_indices

    def recommend_items(self, user_id, max_recommendations, max_neighbours=2, method='user'):
        """
        Recommend items for the given user id
        :param user_id:
        :param max_recommendations:
        :param max_neighbours:
        :param method: 'user' for user-based recommendations, 'item' for item-based
        :return:
        """

        if method == 'user':
            correlations = self.users_correlations
            neighbourhoods = self.users_neighbourhood
            to_check_labels = ['UI matrix', 'users correlations', 'users neighbourhood']
        elif method == 'item':
            correlations = self.items_correlations
            neighbourhoods = self.items_neighbourhood
            to_check_labels = ['UI matrix', 'items correlations', 'items neighbourhood']

        # These variables must not be None
        to_check = [self.ui_matrix, correlations, neighbourhoods]
        for idx, val in enumerate(to_check):
            if val is None:
                return ["Error: Missing " + to_check_labels[idx]]

        user_id_in_ui = user_id - 1
        # for each zero rating, predict ratings
        user_rating_row = self.ui_matrix.getrow(user_id_in_ui)
        zero_idxs = np.where(user_rating_row.toarray() == 0)[1]
        predictions = np.zeros(shape=self.ui_matrix.shape[1])
        for zero_idx in zero_idxs:
            num_neighbours = 0
            correlations_sum = 0
            numerator = 0
            # discriminate user-based recommendations from item-based ones
            if method == 'user':
                # get the neighbours of the user which will get recommendations
                neighbourhood = neighbourhoods[user_id_in_ui]
                for neighbour_id in neighbourhood:
                    # take into account only user neighbours who rated the item
                    # row = the neighbour of the user, column = the item to be automatically rated
                    if self.ui_matrix[neighbour_id, zero_idx] != 0:
                        # row = user which will get recommendations, column = the neighbour of the user
                        correlation = correlations[user_id_in_ui, neighbour_id]
                        if np.isnan(correlation):
                            correlation = 0
                        similarity = correlation
                        correlations_sum += abs(correlation)
                        neighbour_rating = self.ui_matrix[neighbour_id, zero_idx]
                        numerator += similarity * neighbour_rating
                        num_neighbours += 1
                        if num_neighbours == max_neighbours:
                            break
            elif method == 'item':
                # get the neighbours of the item to be rated
                neighbourhood = neighbourhoods[zero_idx]
                for neighbour_id in neighbourhood:
                    # take into account only neighbours rated by the user
                    # row = user which will get recommendations, column = the neighbour of the item to be rated
                    if self.ui_matrix[user_id_in_ui, neighbour_id] != 0:
                        # row = the item to be rated, column = the neighbour of the item to be rated
                        correlation = correlations[zero_idx, neighbour_id]
                        if np.isnan(correlation):
                            correlation = 0
                        similarity = correlation
                        correlations_sum += abs(correlation)
                        neighbour_rating = self.ui_matrix[user_id_in_ui, neighbour_id]
                        numerator += similarity * neighbour_rating
                        num_neighbours += 1
                        if num_neighbours == max_neighbours:
                            break

            denominator = correlations_sum if correlations_sum != 0 else 1
            alpha = 0
            beta = 1
            prediction = alpha + beta * numerator / denominator
            predictions[zero_idx] = prediction

        sorted_indices = sorted(range(len(predictions)), key=lambda i: predictions[i])
        sorted_indices.reverse()
        # increment by one to return the original indices
        recommendations = [x + 1 for x in sorted_indices[:max_recommendations]]
        return recommendations

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
            # increment by one to return the original indices
            recommendations = [x + 1 for x in neighbours[:max_recommendations]]
            return recommendations


