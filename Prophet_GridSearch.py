import numpy as np
import pandas as pd
import warnings
import itertools
import numpy as np
import random
import statsmodels.api as sm
import time
import json
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# prophet by Facebook
from fbprophet import Prophet
# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")

#************* VARIABLES GLOBALES
glb_periodo_training = 3
glb_periodo_futuro = 12

#************* FUNCIONES
def mean_absolute_percentage_error(y_true, y_pred): 
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,4)
    except:
        return None

def run_prophet_base(timeserie):
    try:
        model = Prophet()
        model.fit(timeserie)
        
        forecast = model.make_future_dataframe(periods=glb_periodo_training,freq='MS',include_history=False)
        forecast = model.predict(forecast)
        return forecast
    except:
        return None

def run_prophet_parallel(chunk):
    timeserie = chunk['serie']
    dic_param = chunk['param']
    is_tunning = chunk['is_tunning']

    if is_tunning:
        model =Prophet(
            changepoint_prior_scale = dic_param['changepoint_prior_scale'],                         
            n_changepoints = dic_param['n_changepoints'],
            seasonality_mode = "additive",
            weekly_seasonality=False,
            daily_seasonality = False,
            yearly_seasonality = False, 
            interval_width=0.95
        ).add_seasonality(
            name="adhoc",
            period=365,
            fourier_order = dic_param['fourier_order']
        )
    else:
        model = Prophet()   

    model.fit(timeserie)
    forecast = model.make_future_dataframe(periods=chunk['periodos'],freq='MS',include_history=chunk['with_history'])
    forecast = model.predict(forecast)
    forecast["ID"] = chunk['serie_id']
    
    dic_forecast = {
        'serie_id': chunk['serie_id'],
        'forecast': forecast[ ["ID","ds","yhat","yhat_lower","yhat_upper"] ],
        'param': dic_param
    }
    return dic_forecast


#************* PROGRAMA PRINCIPAL (punto de entrada)
if __name__ == '__main__':
    #raw_path = input('\nRuta de archivo ej. C:\\ruta\\archivo.csv :')
    #max_date = input('\nFecha Maxima total data ej. (incluye esta fecha) 2020-08-01 :')
    #end_date = input('\nFecha fin para training ej. (incluye esta fecha) 2020-05-01 :')    
    raw_path = 'D:\\MY_DOWNLOADS\\CARNOT\\VI_PMM.csv'
    #end_date = '2020-05-01'
    #max_date = '2020-08-01'
    end_date = '2021-01-01'
    max_date = '2021-04-01'    
    
    df = pd.read_csv(raw_path,parse_dates=[0])
    #df["ds"] = pd.to_datetime(df["ds"])

    start_time_global = time.time()
    #recorrer cada ID serie de tiempo
    indexes = df.ID.unique()
    series_base = list()
    contador = 0
    lt_dic_base = list()
    for index in indexes:
        current_df = df.loc[df['ID'] == index]
        
        #training
        mask1 = (current_df['ds'] <= end_date)        
        X_tr = current_df.loc[mask1]
        series_base.append(X_tr)      

        #testing
        mask2 = (current_df['ds'] > end_date)
        X_tst = current_df.loc[mask2]

        my_dict = {"model_index": contador, "series_index": index, "testing":X_tst}
        lt_dic_base.append(my_dict)

        contador += 1

    print('******** INICIA MODELOS BASE ********')
    start_time = time.time()
    p = Pool(cpu_count())
    #recibe solo training
    predictions = list(tqdm(p.imap(run_prophet_base, series_base), total=len(series_base)))
    p.close()
    p.join()
    print("--- %s seconds ----" %(time.time() - start_time))
    print('******** FIN MODELOS BASE ********')

    #calculo de MAPE para modelos base
    print('******** INICIA MAPE - MODELOS BASE ********')
    cnt_modelo_index = 0
    lt_dic_mape_base = list()
    for prediction in predictions:
        var = next( (item for item in lt_dic_base if item["model_index"] == cnt_modelo_index),  None)        
        mape = mean_absolute_percentage_error(var['testing']['y'],prediction['yhat'])
        lt_dic_mape_base.append( {"series_index":var["series_index"],"MAPE_BASE": mape} )        
        cnt_modelo_index += 1
    print('******** FIN MAPE - MODELOS BASE ********')
    

    #CONFIGURACION DE PARAMETROS
    params_grid = {        
        #'seasonality_mode':('multiplicative','additive'),        
        'changepoint_prior_scale':[1,2,3,4,5],        
        'n_changepoints' : [11],
        'fourier_order': [2,5,6]
    }
    grid = ParameterGrid(params_grid)
    cnt = 0
    for p in grid:
        cnt = cnt+1
    print("*******************************************")
    print('\nTotal Possible Models',cnt)    
    print("\n*******************************************")

    print('******** INICIA GRID SEARCH TRAINING ********')
    start_time = time.time()
    lt_parallel_series = list()
    for index in indexes:
        current_df = df.loc[df['ID'] == index]  

        #training
        mask1 = (current_df['ds'] <= end_date)        
        X_tr = current_df.loc[mask1]        

        for p in grid:
            dic_parallel = {
                'is_tunning': True,
                'serie': X_tr,
                'param': {
                    'changepoint_prior_scale': p['changepoint_prior_scale'],
                    'n_changepoints': p['n_changepoints'],
                    'fourier_order': p['fourier_order']
                },
                'serie_id': index,
                'periodos': glb_periodo_training,
                'with_history': False
            }
            lt_parallel_series.append(dic_parallel)    

    #Ejecucion en paralelo de grid search con training
    start_time = time.time()
    p = Pool(cpu_count())    
    grid_predictions = list(tqdm(p.imap(run_prophet_parallel, lt_parallel_series), total=len(lt_parallel_series)))
    p.close()
    p.join()
    print("--- %s seconds ----" %(time.time() - start_time))
    print('******** FIN GRID SEARCH TRAINING ********')    

    model_parameters = pd.DataFrame(columns = ['Series_index','MAPE','Parameters'])
    dict_param = {}
    contador = 0    
    for index in indexes:
        current_df = df.loc[df['ID'] == index]
        Actual = current_df[(current_df['ds']>end_date) & (current_df['ds']<=max_date)]

        filtered_predictions = [d for d in grid_predictions if d['serie_id'] == index]
        for prediction in filtered_predictions:
            grid_prediction = prediction['forecast']

            mape = mean_absolute_percentage_error(Actual['y'],grid_prediction['yhat'])
            dict_param[contador] = {
                'Series_index':index,
                'MAPE':mape,
                'Parameters':prediction['param']
            }
            contador += 1

    df_parametros = model_parameters.from_dict(dict_param, "index")
    df_parametros.to_csv('parametros.csv')

    #MAPE min por serie
    df_min_mape = df_parametros.groupby(['Series_index'])['MAPE'].min().reset_index()

    #Compara MAPE Base vs Grid
    lt_df_compara_mape = list()
    dic_all_mapes = {}    
    for current_dic in lt_dic_mape_base:
        serie_id = current_dic['series_index']
        mape_base = current_dic['MAPE_BASE']
        try:
            mape_grid = df_min_mape.loc[df_min_mape['Series_index'] == serie_id,'MAPE'].values[0]
        except:
            mape_grid = None
        dic_mapes = {'row':[serie_id,mape_base,mape_grid]}
        dic_all_mapes[serie_id] = {'MAPE_BASE': mape_base, 'MAPE_GRID': mape_grid}
        df_compara_mape = pd.DataFrame.from_dict(dic_mapes,orient="index",columns=['ID','MAPE_BASE','MAPE_GRID'])
        lt_df_compara_mape.append(df_compara_mape)

    df_final_compara_mape = pd.concat(lt_df_compara_mape, ignore_index=True)
    df_final_compara_mape.to_csv('MAPE.csv')

    #VERSION PARALELL
    lt_resultados = list()
    lt_parallel_series = list()    
    for index in indexes:
        current_df = df.loc[df['ID'] == index]

        #compara MAPES
        mape_base  = dic_all_mapes[index]['MAPE_BASE']
        mape_grid  = dic_all_mapes[index]['MAPE_GRID']    

        if mape_grid < mape_base:      
            #Modelo con el min MAPE
            condicion_get_modelo = ( (df_parametros['Series_index'] == index) & (df_parametros['MAPE'] == mape_grid) )
            modelo_params = df_parametros.loc[condicion_get_modelo, 'Parameters'].values[0]

            dic_parallel = {
                'is_tunning': True,
                'serie': current_df,
                'param': modelo_params,
                'serie_id': index,
                'periodos': glb_periodo_futuro,
                'with_history': True
            }      
            lt_parallel_series.append(dic_parallel)
        else:
            dic_parallel = {
                'is_tunning': False,
                'serie': current_df,
                'param': None,
                'serie_id': index,
                'periodos': glb_periodo_futuro,
                'with_history': True
            }   
            lt_parallel_series.append(dic_parallel)

    p = Pool(cpu_count())    
    final_predictions = list(tqdm(p.imap(run_prophet_parallel, lt_parallel_series), total=len(lt_parallel_series)))
    p.close()
    p.join()

    for dic_prediction in final_predictions:
        forecast = dic_prediction['forecast']
        lt_resultados.append(forecast)

    resultado_final = pd.concat(lt_resultados, ignore_index=True)
    resultado_final.to_csv('FORECAST.csv')
    print("TOTAL TIME--- %s seconds ----" %(time.time() - start_time_global))