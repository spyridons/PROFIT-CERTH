# PROFIT-CERTH
Code developed in CERTH for the PROFIT project ( http://projectprofit.eu/ )

# recommender_system package
Includes the services for the recommendation system. The code has been developed and tested using Python 3.6.1, 64-bit.

## Instructions
To run the services:
- Rename the "config.ini.template" to "config.ini" and set the appropriate settings in this file (more comments inside the file)
- Run "run_service.py"

## Supported services
### "/recommend_items"
Parameters:
- user_id, the id of the user who will get recommendations
- max_num, the maximum number of recommendations
- method, "item" for item-based collaborative filtering, "user" for user-based collaborative filtering

### "/recommend_users"
Parameters:
- user_id, the id of the user who will get recommendations
- max_num, the maximum number of recommendations

