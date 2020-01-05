# https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv

# =============== Question 1 ======================#
#=========(a)==========
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import statistics
import urllib.request
import matplotlib as mpl
from sklearn.cluster import MiniBatchKMeans, KMeans#Clustering
import shapefile

# download data
# Download the Trip Record Data
# month = 9
# urllib.request.urlretrieve("https://s3.amazonaws.com/nyc-tlc/trip+data/"+ \
#                                "yellow_tripdata_2015-{0:0=2d}.csv".format(month),
 #                               "nyc.2015-{0:0=2d}.csv".format(month))
# Download the location Data
# urllib.request.urlretrieve("https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip", "taxi_zones.zip")

# PATH = "/Users/Kai/Desktop/Project_capitalOne/green_tripdata_2015-09.csv" # local file path where the .csv file is saved.

data = pd.read_csv("/Users/Kai/Dropbox/Project_capitalOne/green_tripdata_2015-09.csv")
# data = pd.read_csv("C:\\Users\\KZ\\Dropbox\\Project_capitalOne\\green_tripdata_2015-09.csv")

# read data into a pandas dataframe
pd.options.display.max_rows = 10
pd.options.display.max_columns = 21
display(data)

# ================（b）======================
numRows = len(data.axes[0]) # 0 for row
numCols = len(data.axes[1]) # 1 for column

print("Number of rows is:", numRows)
print("Number of columns is:", numCols)

#===  data cleaning =====================
# remove those not in new york area ===
data2 = data[(data.Pickup_longitude >= -74.249) & (data.Pickup_latitude >= 40.526) & (data.Pickup_longitude <= -73.909) & (data.Pickup_latitude <= 40.918) \
            & (data.Dropoff_longitude >= -74.249) & (data.Dropoff_latitude >= 40.526) & (data.Dropoff_longitude <= -73.909) & (data.Dropoff_latitude <= 40.918)]

# ============remove those with distance greater than 100 miles=======
ax = data['Trip_distance'].hist(bins=30, figsize=(15,5))
ax.set_yscale('log')
ax.set_xlabel("Trip distance / Miles", fontsize = 14)
ax.set_ylabel("Number of Trips (Log Scale)", fontsize = 14)
plt.title('Histogram of Trip Distances ', fontsize = 14)
plt.show()

data3 = data2[(data2.Trip_distance > 0) & (data2.Trip_distance < 100)]

# ============pickup and drop off locatio on map=============
data['lat_bin'] = np.round(data['Pickup_latitude'],4)
data['lon_bin'] = np.round(data['Pickup_longitude'],4)

coords = data[['Pickup_latitude', 'Pickup_longitude']].values

#Getting 40 clusters using the kmeans
kmeans = MiniBatchKMeans(n_clusters=40, batch_size=10000).fit(coords)
data['Pickup_cluster'] = kmeans.predict(data[['Pickup_latitude', 'Pickup_longitude']])
data['Dropoff_cluster'] = kmeans.predict(data[['Dropoff_latitude', 'Dropoff_longitude']])

#Get regions
def Convert_Clusters(frame,cluster):
    region=[]
    colors = []
    Queens=[4,16,22,23,26,35,36]
    Brooklyn=[11,19,29,37]
    for i in frame[cluster].values:
        if i==2:
            region.append("JFK")
            #Green - JFK
            colors.append('#7CFC00')
        elif i==13:
            region.append("Bronx")
            #Red - Bronx
            colors.append('#DC143C')
        elif i in Queens:
            region.append("Queens")
            #Blue - Queens
            colors.append('#00FFFF')
        elif i in Brooklyn:
            region.append("Brooklyn")
            #Brooklyn - yellow orange
            colors.append('#FFD700')
        else:
            region.append("Manhattan")
            #White - manhattan
            colors.append('#FFFFFF')
    frame['Regions '+cluster]=region
    return frame,colors

data, colors_pickup = Convert_Clusters(data,'Pickup_cluster')
data, colors_dropoff = Convert_Clusters(data,'Dropoff_cluster')
data.head()


def plot_scatter_locations(frame, colors, choose):
    pd.options.display.mpl_style = 'default'  # Better Styling
    new_style = {'grid': False}  # Remove grid
    matplotlib.rc('axes', **new_style)
    rcParams['figure.figsize'] = (17.5, 17)  # Size of figure
    rcParams['figure.dpi'] = 250

    var1 = choose + '_longitude'
    var2 = choose + '_latitude'

    P = frame.plot(kind='scatter', x=[var1], y=[var2], color='white', xlim=(-74.06, -73.77), ylim=(40.61, 40.91), s=.02,
                   alpha=.6)
    P.set_axis_bgcolor('black')  # Background Color
    plt.savefig('plot_' + choose + '.png')
    plt.show()


P = data.plot(kind='scatter', x='Dropoff_longitude', y='Dropoff_latitude', color='white', xlim=(-74.06, -73.77), ylim=(40.61, 40.91), s=.02,
               alpha=.6)
P.set_axis_bgcolor('black')  # Background Color
pd.options.display.mpl_style = 'default'  # Better Styling
new_style = {'grid': False}  # Remove grid
mpl.rc('axes', **new_style)
mpl.rcParams['figure.figsize'] = (17.5, 17)  # Size of figure
mpl.rcParams['figure.dpi'] = 250
plt.savefig('plot_Dropoff.png')
plt.show()


P = data.plot(kind='scatter', x='Pickup_longitude', y='Pickup_latitude', color='white', xlim=(-74.06, -73.77), ylim=(40.61, 40.91), s=.02,
               alpha=.6)
P.set_axis_bgcolor('black')  # Background Color
pd.options.display.mpl_style = 'default'  # Better Styling
new_style = {'grid': False}  # Remove grid
mpl.rc('axes', **new_style)
mpl.rcParams['figure.figsize'] = (17.5, 17)  # Size of figure
mpl.rcParams['figure.dpi'] = 250
plt.savefig('plot_Pickup.png')
mpl.rcParams['figure.dpi'] = 250
plt.show()

################# Question 2 ##########################
# ============== (a) ====================
display(data.Trip_distance) # use the display_data helper function to view the 5 first rows
minTripDistance = data['Trip_distance'].min() # minimum trip distance
maxTripDistance = data['Trip_distance'].max() # maximum trip distance
print("The minimum trip distance is: ", minTripDistance, " miles")
print("The maximum trip distance is: ", maxTripDistance, " miles")

# miximum distance 603.1miles? clean it
tripDistanceCount = data['Trip_distance'].value_counts() # trip distance value count
display(tripDistanceCount)

# It seems most of the trip distance are within the range of [0,30]
maxDistShow = 100
numBins = 100
plt.hist(data['Trip_distance'][data['Trip_distance']<maxDistShow],bins = numBins)
plt.title('Histogram of Trip Distances ( < 100 miles )', fontsize = 14)
plt.xlabel('Trip Distance', fontsize = 14)
plt.ylabel('Number of Trips', fontsize = 14)
plt.show()


# =================== (b) =================
# Find the number of trips with distance in [0,5], (6, 10] and (10, 15]
# data[5 < data['Trip_distance'] && data['Trip_distance'] <= 10]['Trip_distance'].count() # trip distance value count
data.groupby(pd.cut(data['Trip_distance'], np.arange(0,20,5))).size()


# ################### Question 3 ########################
# we're stripping the hour field from the pickup_datetime field to create a new field named "pickup_hour"
data['pickup'] = pd.to_datetime(data['lpep_pickup_datetime'], format = '%Y-%m-%d %H:%M:%S')# apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
data['pickup_hour'] = data['pickup'].apply(lambda x: x.hour) # this is a new field for holding the pickup_hour.

# we're stripping the hour field from the dropoff_datetime field to create a new field named "dropoff_hour"
data['dropoff'] = pd.to_datetime(data['Lpep_dropoff_datetime'], format = '%Y-%m-%d %H:%M:%S')
data['dropoff_hour'] = data['dropoff'].apply(lambda x: x.hour) # this is a new field for holding the dropoff_hour.

data[['Trip_distance', 'pickup_hour']].groupby('pickup_hour').mean().plot.bar()
plt.title('Mean Trip Distance in One Day', fontsize = 14)
plt.xlabel('Pickup Time / 24H', fontsize = 14)
plt.ylabel('Trip Distance / Miles', fontsize = 14)
plt.show()

data[['Trip_distance','pickup_hour']].groupby('pickup_hour').median().plot.bar()
plt.title('Median Trip Distance in One Day', fontsize = 14)
plt.xlabel('Pickup Time / 24H', fontsize = 14)
plt.ylabel('Trip Distance / Miles', fontsize = 14)
plt.show()

# =========(b)============
# Define a function to tell whether the pickup/dropoff point is at an airport.
airportPos = {'JFK': [[40.646677, 40.666467], [-73.821884, -73.750296]], 'LGA': [[40.767550, 40.773098],[-73.884547, -73.865387]]}
def findAirport(row):
    if (((row['Pickup_longitude'] < airportPos['JFK'][1][1]) & (row['Pickup_longitude'] > airportPos['JFK'][1][0]) &  # we got this directions from google maps
        (row['Pickup_latitude'] < airportPos['JFK'][0][1]) & (row['Pickup_latitude'] > airportPos['JFK'][0][0])) |
        ((row['Dropoff_longitude'] < airportPos['JFK'][1][1]) & (row['Dropoff_longitude'] > airportPos['JFK'][1][0]) &
        (row['Dropoff_latitude'] < airportPos['JFK'][0][1]) & (row['Dropoff_latitude'] > airportPos['JFK'][0][0]))):
        return 'JFK'  # John F. Kennedy International Airport
    if (((row['Pickup_longitude'] < airportPos['LGA'][1][1]) & (row['Pickup_longitude'] > airportPos['LGA'][1][0]) &  # long and lat from google maps
        (row['Pickup_latitude'] < airportPos['LGA'][0][1]) & (row['Pickup_latitude'] > airportPos['LGA'][0][0])) |
        ((row['Dropoff_longitude'] < airportPos['LGA'][1][1]) & (row['Dropoff_longitude'] > airportPos['LGA'][1][0]) &
        (row['Dropoff_latitude'] < airportPos['LGA'][0][1]) & (row['Dropoff_latitude'] > airportPos['LGA'][0][0]))):
        return 'LAG'  # LaGuardia Airport
    else:
        return 'NOT'  # Not an Airport pickup/dropoff

# this is to create a new field in the dataframe based on the helper function written above.
data['Airport'] = data.apply(findAirport, axis=1)

# what's the distribution of the rides.
data['Airport'].value_counts()
print('Average Fair for Trips To/From Airport: ', data[data['Airport']!= 'NOT']['Fare_amount'].mean()) #average fare
print('Number of Trips To/From Airport: ', data[data['Airport']!='NOT']['Fare_amount'].shape[0])


#============== my question 1: how many customers during the day ===========================
# how many customers in each hour in JFK airport ================
data[data['Airport']== 'JFK'].groupby(['pickup_hour']).size().plot.bar() #
plt.title('Number of Customers During A Day From/To John Kennedy Airport', fontsize = 14)
plt.xlabel('Pickup Time / 24H', fontsize = 14)
plt.ylabel('Number of Customers', fontsize = 14)
plt.show()

# how many customers in each hour in LAG airport
data[data['Airport']== 'LAG'].groupby(['pickup_hour']).size().plot.bar() #
plt.title('Number of Customers During A Day From/To LaGuardia Airport', fontsize = 14)
plt.xlabel('Pickup Time / 24H', fontsize = 14)
plt.ylabel('Number of Customers', fontsize = 14)
plt.show()

# how many customers in each hour in Non airport airport
data[data['Airport']== 'NOT'].groupby(['pickup_hour']).size().plot.bar() #
plt.title('Number of Customers During A Day From & To Non-airport Area', fontsize = 14)
plt.xlabel('Pickup Time / 24H', fontsize = 14)
plt.ylabel('Number of Customers', fontsize = 14)
plt.show()


# =========== By each day of a week ==================
data['pickup_day'] = pd.to_datetime(data['lpep_pickup_datetime'], format = '%Y-%m-%d %H:%M:%S').apply(lambda x: x.day) # this is a new field for holding the pickup_hour.
data['dropoff_day'] = pd.to_datetime(data['Lpep_dropoff_datetime'], format = '%Y-%m-%d %H:%M:%S').apply(lambda x: x.day) # this is a new field for holding the pickup_hour.

hour_pickups = []
temp = []
for i in range(1,8):
    for j in range(0,24):
        temp.append(data[(data.pickup_day == i) & (data.pickup_hour == j) & (data['Airport'] == 'NOT')].shape[0])
    hour_pickups.append(temp)
    temp = []
colors = ['xkcd:blue','xkcd:orange','xkcd:brown','xkcd:coral','xkcd:magenta','xkcd:green','xkcd:fuchsia']
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

plt.figure(figsize=(8,4))
hours_lis = [s for s in range(0,24)]
for k in range(0,7):
    plt.plot(hours_lis,hour_pickups[k],colors[k],label = days[k])
    plt.plot(hours_lis,hour_pickups[k], 'ro',  markersize=2)

plt.xticks([s for s in range(0,24)])
plt.xlabel('Hours of a day')
plt.ylabel('Number of pickups')
plt.title('Pickups for every hour')
plt.legend()
plt.grid(True)
plt.show()

# ================== my question 2: passenger number for each airport =================
# how many passengers on each trip (To/from JFK airport)  ================
data[data['Airport']== 'JFK'].groupby(['Passenger_count']).size().plot.bar() #
plt.title('Number of Passengers on Each Trip From/To John Kennedy Airport', fontsize = 14)
plt.xlabel('Trips Number', fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.show()

data[data['Airport']== 'LAG'].groupby(['Passenger_count']).size().plot.bar() #
plt.title('Number of Passengers on Each Trip From/To LaGuardia Airport', fontsize = 14)
plt.xlabel('Trips Number', fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.show()

data[data['Airport']== 'NOT'].groupby(['Passenger_count']).size().plot.bar() #
plt.title('Number of Passengers on Each Trip From & To Non-airport Area', fontsize = 14)
plt.xlabel('Trips Number', fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.show()

# =================== my questions 3: what payment types people usually use? ===========
"""Payment types:
1. Credit card
2. Cash
3. No charge
4. Dispute
5. Unknown
6. Voided trip
"""
# This is interesting, I'm surprised most people choose to pay by cash
payment_uniq = set(data['Payment_type'].values)
payment_count = []
for i in payment_uniq :
    payment_count.append(data[data['Payment_type'] == i].shape[0])
print (payment_count)
payment_uniq = list(payment_uniq)
print (payment_uniq)

pay_pickups = [x for _,x in sorted(zip(payment_uniq,payment_count))]
pay = [int(y) for y,_ in sorted(zip(payment_uniq,payment_count))]
pay_type = ['Credit card','Cash','No charge','Dispute','Unknow']

sns.set_style("whitegrid")

fig, ax = plt.subplots()
fig.set_size_inches(12, 5)

ax = sns.barplot(x = np.array(pay_type) , y = np.array(pay_pickups))
ax.set(ylabel='Pickups',xlabel = 'payment types')
sns.plt.title('Number of pickups for each payment types')
plt.show()

# =================== my question 4: Among the trips payed by cache, does passenger number influence tip amount? =========
data.groupby(pd.cut(data['Tip_amount'], np.arange(-1, 30, 2)))['Tip_amount'].count().plot.bar()
plt.title('Number of Passengers on Each Trip From/To LaGuardia Airport', fontsize = 14)
plt.xlabel('Trips Number', fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.show()


#data2 = data[data['Payment_type'] == 2] # by cash, nobody pays cash, this is not weird, I guess driver just does not report their tips
data2 = data[(data['Payment_type'] == 2)]
data2.groupby(pd.cut(data2['Tip_amount'], np.arange(-1, 30, 2)))['Tip_amount'].size().plot.bar()
plt.title('Number of Trips For Each Tip Amount (Pay with Cash)', fontsize = 14)
plt.xlabel('Tip Amount', fontsize = 14)
plt.ylabel('Number of Trips', fontsize = 14)
plt.show()


data3 = data[data['Payment_type'] == 1] # by credit card, tip information just uploads automatically
#data2 = data[(data['Payment_type'] == 2) & (data['Passenger_count'] == 1)]
data3.groupby(pd.cut(data3['Tip_amount'], np.arange(-1, 30, 2)))['Tip_amount'].size().plot.bar()
plt.title('Number of Passengers on Each Trip From/To LaGuardia Airport', fontsize = 14)
plt.xlabel('Trips Number', fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.show()

data4 = data[(data['Payment_type'] == 1) & (data['Passenger_count'] == 10)] # by credit card, tip information just uploads automatically
#data2 = data[(data['Payment_type'] == 2) & (data['Passenger_count'] == 1)]
data4.groupby(pd.cut(data4['Tip_amount'], np.arange(-1, 30, 2)))['Tip_amount'].size().plot.bar()
plt.title('Number of Passengers on Each Trip From/To LaGuardia Airport', fontsize = 14)
plt.xlabel('Trips Number', fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.show()

########################### Question 4 ######################
data['tip_percent'] = (data['Tip_amount']/data['Total_amount']).apply(lambda x: x * 100) # calculate tip percentage

grouped_df = data.groupby(['pickup_hour', 'Airport'])['tip_percent'].aggregate(np.mean).reset_index() #average tip

plt.figure(figsize=(12, 8))
sns.pointplot(grouped_df.pickup_hour.values, grouped_df.tip_percent.values, grouped_df.Airport.values, alpha=0.8)
plt.ylabel('Average Tip Percentage of the Fare / USD', fontsize = 14)
plt.xlabel('Pick-Up Time / 24H', fontsize = 14)
plt.title('Tip Amount During A Day', fontsize = 14)
plt.xticks(rotation='vertical', fontsize = 14)
plt.show()


# ============= (b) ==================
import xgboost as xgb
import lightgbm as lgb

df_model = pd.read_csv("/Users/Kai/Dropbox/Project_capitalOne/green_tripdata_2015-09.csv", parse_dates=['lpep_pickup_datetime'])
#df_model = pd.read_csv("C:\\Users\\KZ\\Dropbox\\Project_capitalOne\\green_tripdata_2015-09.csv",  parse_dates=['lpep_pickup_datetime'])

df_model = df_model.reset_index() # reset the index of our dataframe.
df_model.rename(columns={'index': 'ID'}, inplace=True) # my

df_model.drop(['Ehail_fee', 'RateCodeID', 'Extra'], axis=1, inplace=True) #drop these features, because they're either all 0's or Nan's
display(df_model) # let's see how our data looks like. The ID field is helpful during the prediction stage)

df_model['tip_percent'] = df_model['Tip_amount']/df_model['Total_amount'] # calculate the tip percent
df_model['tip_percent'] = df_model['tip_percent'].apply(lambda x: x * 100) # multiply the value by 100
df_model = df_model[df_model['Tip_amount'] > 0] # make sure the tip is greater than zero
df_model = df_model[df_model['Fare_amount'] > 0] # make sure the fare amount is greater than zero

# Train-test split
y = df_model.tip_percent # tip_percent is our target variable
X = df_model # predictor varibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # test size = 0.2, hence 80-20 split.
print ( "\nX_train:\n")
print (X_train.shape)
print ("\nX_test:\n")
print (X_test.shape)

X_train.head() # this is our training data

X_test = X_test.iloc[:, :-1] # exclude the target variable(tip_percent) from our test data

X_test.shape # test data shape
X_test.head() # test data

# it's also necessary to drop the tip_amount, because then it'd be easy for the model to identify the percentage
# of tip by just dividing it with the total fare.
X_test.drop(['Tip_amount'], axis=1, inplace=True)

X_train['log_tip_percent'] = np.log1p(X_train['tip_percent'].values) # logarithm of the tip_percent, because we use RMSLE

plt.figure(figsize=(8,6))
plt.scatter(range(X_train.shape[0]), np.sort(X_train.tip_percent.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('log_tip_percent', fontsize=12)
plt.show()

null_count_df = X_train.isnull().sum(axis=0).reset_index() # training set
null_count_df.columns = ['col_name', 'null_count']
null_count_df

null_count_df = X_test.isnull().sum(axis=0).reset_index() # test set
null_count_df.columns = ['col_name', 'null_count']
null_count_df


X_train['pickup_date'] = X_train['lpep_pickup_datetime'].dt.date
X_test['pickup_date'] = X_test['lpep_pickup_datetime'].dt.date

"""
cnt_srs = X_train['pickup_date'].value_counts() # train pickup date
plt.figure(figsize=(12,4))
ax = plt.subplot(111)
ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='vertical')
plt.show()

cnt_srs = X_test['pickup_date'].value_counts() # test pickup date
plt.figure(figsize=(12,4))
ax = plt.subplot(111)
ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='vertical')
plt.show()
"""
# day of the month
X_train['pickup_day'] = X_train['lpep_pickup_datetime'].dt.day
X_test['pickup_day'] = X_test['lpep_pickup_datetime'].dt.day

# month of the year
X_train['pickup_month'] = X_train['lpep_pickup_datetime'].dt.month
X_test['pickup_month'] = X_test['lpep_pickup_datetime'].dt.month

# hour of the day
X_train['pickup_hour'] = X_train['lpep_pickup_datetime'].dt.hour
X_test['pickup_hour'] = X_test['lpep_pickup_datetime'].dt.hour

# Week of year
X_train["week_of_year"] = X_train["lpep_pickup_datetime"].dt.weekofyear
X_test["week_of_year"] = X_test["lpep_pickup_datetime"].dt.weekofyear

# Day of week
X_train["day_of_week"] = X_train["lpep_pickup_datetime"].dt.weekday
X_test["day_of_week"] = X_test["lpep_pickup_datetime"].dt.weekday

# Convert to numeric
map_dict = {'N':0, 'Y':1}
X_train['Store_and_fwd_flag'] = X_train['Store_and_fwd_flag'].map(map_dict)
X_test['Store_and_fwd_flag'] = X_test['Store_and_fwd_flag'].map(map_dict)

# drop off the variables which are not needed
cols_to_drop = ['ID', 'lpep_pickup_datetime', 'pickup_date', 'Lpep_dropoff_datetime']
train_id = X_train['ID'].values
test_id = X_test['ID'].values
train_y = X_train.log_tip_percent.values
train_X = X_train.drop(cols_to_drop + ['Tip_amount', 'tip_percent', 'log_tip_percent'], axis=1)
test_X = X_test.drop(cols_to_drop, axis=1)

def runXGB(train_X, train_y, val_X, val_y, test_X, eta=0.05, max_depth=5, min_child_weight=1, subsample=0.8, colsample=0.7, num_rounds=8000, early_stopping_rounds=100, seed_val=2017):
    params = {}
    params["objective"] = "reg:linear"
    params['eval_metric'] = "rmse"
    params["eta"] = eta
    params["min_child_weight"] = min_child_weight
    params["subsample"] = subsample
    params["colsample_bytree"] = colsample
    params["silent"] = 1
    params["max_depth"] = max_depth
    params["seed"] = seed_val
    params["nthread"] = -1

    plst = list(params.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgval = xgb.DMatrix(val_X, label = val_y)
    xgtest = xgb.DMatrix(test_X)
    watchlist = [ (xgtrain,'train'), (xgval, 'test') ]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=20)

    pred_val = model.predict(xgval, ntree_limit=model.best_ntree_limit)
    pred_test = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

    return pred_val, pred_test

def runLGB(train_X, train_y, val_X, val_y, test_X, eta=0.05, num_leaves=10, max_depth=5, min_child_weight=1, subsample=0.8, colsample=0.7, num_rounds=8000, early_stopping_rounds=100, seed_val=2017):
    params = {}
    params["objective"] = "regression"
    params['metric'] = "l2_root"
    params["learning_rate"] = eta
    params["min_child_weight"] = min_child_weight
    params["bagging_fraction"] = subsample
    params["bagging_seed"] = seed_val
    params["feature_fraction"] = colsample
    params["verbosity"] = 0
    params["max_depth"] = max_depth
    params["num_leaves"] = num_leaves
    params["nthread"] = -1

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label = val_y)
    model = lgb.train(params, lgtrain, num_rounds, valid_sets=lgval, early_stopping_rounds=early_stopping_rounds, verbose_eval=20)

    pred_val = model.predict(val_X, num_iteration=model.best_iteration)
    pred_test = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_val, pred_test, model

from sklearn import model_selection, preprocessing, metrics # import a few other modules


kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
cv_scores = []
pred_test_full = 0
pred_val_full = np.zeros(X_train.shape[0])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.ix[dev_index], train_X.ix[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val, pred_test, model = runLGB(dev_X, dev_y, val_X, val_y, test_X, num_rounds=5000, num_leaves=10, max_depth=8, eta=0.3)
    pred_val_full[val_index] = pred_val
    pred_test_full += pred_test
    cv_scores.append(np.sqrt(metrics.mean_squared_error(val_y, pred_val)))
print(cv_scores)
print("Mean RMSE score : ",np.mean(cv_scores))

pred_test_full = pred_test_full / 5.
pred_test_full = np.expm1(pred_test_full)
pred_val_full = np.expm1(pred_val_full)

# saving train predictions for ensemble #
train_pred_df = pd.DataFrame({'ID':train_id})
train_pred_df['tip_percent'] = pred_val_full
train_pred_df.to_csv("train_preds_lgb_baseline.csv", index=False)

# saving test predictions for ensemble #
test_pred_df = pd.DataFrame({'ID':test_id})
test_pred_df['tip_percent'] = pred_test_full
test_pred_df.to_csv("test_preds_lgb_baseline.csv", index=False)

print('Plot feature importances...')
ax = lgb.plot_importance(model, max_num_features=10, height = 0.5)
plt.show()

# ================= Question 5 =========================
data.head()
ans_t = (data['dropoff'] - data['pickup']).apply(lambda x: x.total_seconds()) # extract the seconds from pickup data
print('Percentage of entries with travel time less than a minute: ',100 * data[ans_t < 60].shape[0]/data.shape[0],'%')
data['travel_time'] = (data['dropoff'] - data['pickup']).apply(lambda x: x.total_seconds())
data = data[data['travel_time'] > 60] # travel time greater than 60 seconds
data['average_speed'] = 3600*(data['Trip_distance']/data['travel_time'])

data['average_speed'].plot.hist(bins=10)
plt.show()


print('No of entries with average speed over 100 miles per hour: ',(data['average_speed']>100).value_counts()[1])
data = data[data['average_speed']<100]

# (b) Can you perform a test to determine if the average trip speeds are materially the same in all weeks of September? If you decide they are not the same, can you form a hypothesis regarding why they differ?

data['week'] = data['dropoff'].apply(lambda x: x.week) # extract week of year

data['week'].value_counts() # week count
week_1 = data['average_speed'][data['week']==36].as_matrix() # reassign week=36 to week_1 df
week_2 = data['average_speed'][data['week']==37].as_matrix() # reassign week=37 to week_2 df
week_3 = data['average_speed'][data['week']==38].as_matrix() # reassign week=38 to week_3 df
week_4 = data['average_speed'][data['week']==39].as_matrix() # reassign week=39 to week_4 df
week_5 = data['average_speed'][data['week']==40].as_matrix() # reassign week=40 to week_5 df

stats.f_oneway(week_1,week_2, week_3,week_4, week_5)


print(week_1.mean(),week_2.mean(),week_3.mean(),week_4.mean(),week_5.mean())

print(statistics.median(week_1),statistics.median(week_2),statistics.median(week_3),statistics.median(week_4),
      statistics.median(week_5))

plt.rcParams["figure.figsize"] = [20,12]
plt.subplot(3,2,1)
plt.hist(week_1,bins = 50,label = 'week 1')
plt.legend()
plt.subplot(3,2,2)
plt.hist(week_2,bins = 50,label = 'week 2')
plt.legend()
plt.subplot(3,2,3)
plt.hist(week_3,bins = 50,label = 'week 3')
plt.legend()
plt.subplot(3,2,4)
plt.hist(week_4,bins = 50,label = 'week 4')
plt.legend()
plt.subplot(3,2,5)
plt.hist(week_5,bins = 50,label = 'week 5')
plt.legend()
plt.legend()
plt.savefig('task5')
plt.show()

grouped = data.groupby('pickup_hour') # group by the hour
samples = []

for name,group in grouped:
    samples.append(group['average_speed']) # append the avg speed data


sample = samples
stats.f_oneway(sample[0],sample[1],sample[2],sample[3], sample[4],sample[5],sample[6],sample[7],sample[8],sample[9],
              sample[10],sample[11],sample[12],sample[13],sample[14],sample[15],sample[16],sample[17],sample[18],
               sample[19],
              sample[20],sample[21],sample[22],sample[23])

means = [] # empty list for storing the mean info
medians = [] # empty list for storing the median info
for hour in range(24):
    means.append(statistics.mean(sample[hour]))
    print('Mean:',statistics.mean(sample[hour]))
    medians.append(statistics.median(sample[hour]))
    print('Median:',statistics.median(sample[hour]))

plt.style.use('ggplot')

plt.bar(range(1,25), means, color='blue')
plt.xlabel("Hour of Day", fontsize=15)
plt.ylabel("Average Speed (mi/hr)", fontsize=15)
plt.title("Average speed of a ride at the hour")

plt.show()
