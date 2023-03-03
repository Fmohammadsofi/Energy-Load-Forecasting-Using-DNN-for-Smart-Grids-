%% Load Forecasting Demo
% This demo uses a neural network to make predictions for the electrical
% load on a zone in New York, based on weather data and historical load data.
% 
% The load data was downloaded from the New York ISO
% (http://mis.nyiso.com/public/) and the weather data was downloaded from
% the National Climatic Data Center
% (http://cdo.ncdc.noaa.gov/qclcd_ascii/). Both the load and weather data
% went through an aggregation and cleaning process before being saved as
% mat files. For information on how the data was cleaned see the recorded
% webinar "Data Analytics with MATLAB"
% 
% Copyright 2015 The MathWorks, Inc.

%% Load and joing the clean data
load nyiso_cleaned
load weather_cleaned
%load householdpowerconsumptionhourly
% join together the nyiso and weatherData tables keeping only the
% timestamps that exist in both data sets
nyiso = innerjoin(nyiso, weatherData);

%% Modeling a single zone
% Now we're ready to start modeling our data so we can make forecasts. The
% rest of the script examines a neural network applied to the New York
% City area. Full example in Script8_Modeling

% Pull out relevant data into separate table to make it easier to work with
modeldata = nyiso(:,{'Date','N_Y_C_','TemperatureKLGA'});
% Change the table column names to be more general
modeldata.Properties.VariableNames(2:3) = {'Load','Temperature'};

%% Create predictors
% In order to build an accurate model, we need useful predictors to work
% with. A common technique with temporal predictors is to break them into
% their separate parts so they can be varied independently of each other.

% Create temporal predictors
modeldata.Hour = modeldata.Date.Hour;
modeldata.Month = modeldata.Date.Month;
modeldata.Year = modeldata.Date.Year;
modeldata.DayOfWeek = weekday(modeldata.Date);
modeldata.isWeekend = ismember(modeldata.DayOfWeek,[1,7]);

% Pull the temperature and dew point apart into separate columns as
% expected by the machine learning techniques
modeldata.Temp = modeldata.Temperature(:,1);
modeldata.DewPnt = modeldata.Temperature(:,2);
modeldata.Temperature = [];

%% Lagged predictors
% The load data itself can be used as a predictor. We could use a
% traditional time series analysis, but a look at the autocorrelation will
% show an interesting pattern that suggest use of lagged predictors of 1
% and 7 days

% Compute and plot autocorrelation in load data
[c,lags] = xcorr(modeldata.Load(~isnan(modeldata.Load)),200);
plot(lags/24,c) 
xlim([0,200/24]) 

% Create predictor for the load at the same time the prior day, 24 hour
% lag. Because we have missing timestamps, we can't simply look back by 24
% rows. This method is robust to missing timestamps.
modeldata.PriorDay = nan(height(modeldata),1);
idxload = ismember(modeldata.Date+hours(24),modeldata.Date);
idxprior = ismember(modeldata.Date-days(1),modeldata.Date);
modeldata.PriorDay(idxprior) = modeldata.Load(idxload);

% Create predictor for the load at the same time the prior hour, 1 hour
% lag. Because we have missing timestamps, we can't simply look back by 1
% row. This method is robust to missing timestamps.
modeldata.PriorHour = nan(height(modeldata),1);
idxload = ismember(modeldata.Date+hours(1),modeldata.Date);
idxprior = ismember(modeldata.Date-hours(1),modeldata.Date);
modeldata.PriorHour(idxprior) = modeldata.Load(idxload);

% Create predictor for the load at the same time the same day the prior
% week, 168 hour lag
modeldata.PriorWeek = nan(height(modeldata),1);
idxload = ismember(modeldata.Date+hours(168),modeldata.Date);
idxprior = ismember(modeldata.Date-days(7),modeldata.Date);
modeldata.PriorWeek(idxprior) = modeldata.Load(idxload);

%% Split to training and testing
%
% Here we split the data into training and testing sets based on the date.
% More sophisticated techniques such as cross validation are also
% available.
idxtrain = modeldata.Date <= datetime('31-May-2012','TimeZone','America/New_York');
idxtest = modeldata.Date > datetime('31-May-2012','TimeZone','America/New_York');
% some of the machine learning functions expect separate matricies for the
% inputs and output
%Xvars = {'Hour','Month','Year','DayOfWeek','isWeekend','Temp','DewPnt','PriorDay','PriorWeek'};
Xvars = {'Hour','Month','DayOfWeek','isWeekend','Temp','DewPnt','PriorDay','PriorHour','PriorWeek'};

Yvars = {'Load'};
Xtrain = modeldata{idxtrain,Xvars}';
Ytrain = modeldata{idxtrain,Yvars}';
Xtest = modeldata{idxtest,Xvars}';
Ytest = modeldata{idxtest,Yvars}';

%% Neural network tool
nftool
trainFcn = 'trainlm';  % Levenberg-Marquardt training algorithm

% Create a Fitting Network
hiddenLayerSize = 20; 
net = fitnet(hiddenLayerSize,trainFcn);

%% make predictions
% make predictions on the test data set and plot results
Y_nn = myNeuralNetworkFunction(Xtest);
figure
ax1=subplot(2,1,1);
plot(modeldata.Date(idxtest),Y_nn','DisplayName','Y_nn');hold on
plot(modeldata.Date(idxtest),Ytest','DisplayName','Ytest');hold off
legend('Neural Network','Measured')
ylabel('Load (MW)')
%Y_nn_train = myNeuralNetworkFunction(Xtrain);
ax2=subplot(2,1,2);
plot(modeldata.Date(idxtest),(Ytest'-Y_nn'),'.');
legend('Neural Network')
ylabel('Error (MW)')
linkaxes([ax1,ax2],'x')


%% make predictions
% make predictions on the test data set and plot results
Y_nn = net(Xtest);
figure
ax1=subplot(2,1,1);
plot(modeldata.Date(idxtest),Y_nn','DisplayName','Y_nn');hold on
plot(modeldata.Date(idxtest),Ytest','DisplayName','Ytest');hold off
legend('Neural Network','Measured')
ylabel('Load (MW)')
ax2=subplot(2,1,2);
plot(modeldata.Date(idxtest),Ytest'-Y_nn');
legend('Neural Network')
ylabel('Error (MW)')
linkaxes([ax1,ax2],'x')

